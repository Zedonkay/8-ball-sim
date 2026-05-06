#ifdef __APPLE__
#define GL_SILENCE_DEPRECATION
#include <GLUT/glut.h>
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#else
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace
{

constexpr float    SPHERE_R       = 0.15f;
constexpr uint32_t FRAME_MAGIC_V1 = 0x464D4953u; // SIMF
constexpr uint32_t FRAME_MAGIC_V2 = 0x464D4932u; // SIM2
constexpr int      MAX_REASONABLE_PARTICLES = 2000000;

struct Frame
{
    int32_t step    = 0;
    float   t       = 0.0f;
    int32_t n_fluid = 0;
    int32_t n_total = 0;

    std::vector<float> x;
    std::vector<float> y;
    std::vector<float> z;
    std::vector<float> density;

    bool  has_die = false;
    float die_pos[3]  = {0.0f, 0.0f, 0.0f};
    float die_quat[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    float die_half[3] = {0.0f, 0.0f, 0.0f};
};

std::mutex              g_frame_mutex;
std::shared_ptr<Frame>  g_latest_frame;
std::string             g_stream_error;
std::atomic<bool>       g_stream_done{false};
std::atomic<uint64_t>   g_stream_frames{0};
std::atomic<uint64_t>   g_last_stream_ns{0};
uint64_t                g_version = 0;

int   g_width         = 1100;
int   g_height        = 850;
float g_yaw           = 35.0f;
float g_pitch         = 18.0f;
float g_distance      = 0.48f;
float g_point_size    = 5.0f;
float g_target_fps    = 60.0f;
int   g_max_particles = 0;
int   g_max_ghosts    = 1200;
bool  g_show_ghosts   = false;

bool g_dragging = false;
int  g_last_mouse_x = 0;
int  g_last_mouse_y = 0;

double g_render_fps = 0.0;
double g_stream_fps = 0.0;

uint64_t steady_now_ns()
{
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

bool read_exact(std::istream &in, void *dst, std::size_t bytes)
{
    in.read(static_cast<char *>(dst), static_cast<std::streamsize>(bytes));
    return static_cast<std::size_t>(in.gcount()) == bytes;
}

template <typename T>
bool read_value(std::istream &in, T &value)
{
    return read_exact(in, &value, sizeof(T));
}

bool valid_counts(int32_t n_fluid, int32_t n_total)
{
    return n_fluid >= 0 && n_total >= 0 && n_fluid <= n_total &&
           n_total <= MAX_REASONABLE_PARTICLES;
}

bool read_frame(std::istream &in, Frame &frame, std::string &error)
{
    uint32_t magic = 0;
    if(!read_value(in, magic))
        return false;

    if(magic != FRAME_MAGIC_V1 && magic != FRAME_MAGIC_V2)
    {
        std::ostringstream ss;
        ss << "bad frame magic 0x" << std::hex << magic
           << "; the stream is not aligned binary SIM data";
        error = ss.str();
        return false;
    }

    if(!read_value(in, frame.step) ||
       !read_value(in, frame.t) ||
       !read_value(in, frame.n_fluid) ||
       !read_value(in, frame.n_total))
    {
        return false;
    }

    if(!valid_counts(frame.n_fluid, frame.n_total))
    {
        std::ostringstream ss;
        ss << "bad particle counts: n_fluid=" << frame.n_fluid
           << " n_total=" << frame.n_total;
        error = ss.str();
        return false;
    }

    frame.x.resize(frame.n_total);
    frame.y.resize(frame.n_total);
    frame.z.resize(frame.n_total);
    frame.density.resize(frame.n_fluid);

    const std::size_t total_bytes =
        static_cast<std::size_t>(frame.n_total) * sizeof(float);
    const std::size_t fluid_bytes =
        static_cast<std::size_t>(frame.n_fluid) * sizeof(float);

    if(!read_exact(in, frame.x.data(), total_bytes) ||
       !read_exact(in, frame.y.data(), total_bytes) ||
       !read_exact(in, frame.z.data(), total_bytes) ||
       !read_exact(in, frame.density.data(), fluid_bytes))
    {
        return false;
    }

    frame.has_die = magic == FRAME_MAGIC_V2;
    if(frame.has_die)
    {
        if(!read_exact(in, frame.die_pos, sizeof(frame.die_pos)) ||
           !read_exact(in, frame.die_quat, sizeof(frame.die_quat)) ||
           !read_exact(in, frame.die_half, sizeof(frame.die_half)))
        {
            return false;
        }
    }

    return true;
}

void reader_loop()
{
    while(true)
    {
        auto frame = std::make_shared<Frame>();
        std::string error;
        if(!read_frame(std::cin, *frame, error))
        {
            std::lock_guard<std::mutex> lock(g_frame_mutex);
            g_stream_error = error;
            g_stream_done = true;
            return;
        }

        {
            std::lock_guard<std::mutex> lock(g_frame_mutex);
            g_latest_frame = std::move(frame);
            ++g_version;
        }
        ++g_stream_frames;
        g_last_stream_ns.store(steady_now_ns());
    }
}

float clamp01(float x)
{
    return std::max(0.0f, std::min(1.0f, x));
}

int stride_for(int n, int max_n)
{
    if(max_n <= 0 || n <= max_n)
        return 1;
    return std::max(1, (n + max_n - 1) / max_n);
}

void color_for_density(float rho)
{
    const float u = clamp01((rho - 800.0f) / 500.0f);
    glColor4f(0.10f + 0.85f * u,
              0.35f + 0.45f * std::min(1.0f, u * 1.4f),
              0.95f - 0.70f * u,
              0.78f);
}

void draw_sphere()
{
    glColor4f(0.45f, 0.62f, 0.75f, 0.26f);
    glLineWidth(1.0f);

    constexpr int slices = 48;
    constexpr int rings = 10;

    for(int r = 1; r <= rings; ++r)
    {
        const float phi = 3.14159265f * static_cast<float>(r) /
                          static_cast<float>(rings + 1);
        glBegin(GL_LINE_LOOP);
        for(int i = 0; i < slices; ++i)
        {
            const float theta = 2.0f * 3.14159265f *
                                static_cast<float>(i) /
                                static_cast<float>(slices);
            glVertex3f(SPHERE_R * std::sin(phi) * std::cos(theta),
                       SPHERE_R * std::sin(phi) * std::sin(theta),
                       SPHERE_R * std::cos(phi));
        }
        glEnd();
    }

    for(int m = 0; m < 8; ++m)
    {
        const float theta = 2.0f * 3.14159265f *
                            static_cast<float>(m) / 8.0f;
        glBegin(GL_LINE_STRIP);
        for(int i = 0; i <= slices / 2; ++i)
        {
            const float phi = 3.14159265f * static_cast<float>(i) /
                              static_cast<float>(slices / 2);
            glVertex3f(SPHERE_R * std::sin(phi) * std::cos(theta),
                       SPHERE_R * std::sin(phi) * std::sin(theta),
                       SPHERE_R * std::cos(phi));
        }
        glEnd();
    }
}

void quat_to_matrix(const float q[4], float r[9])
{
    float w = q[0], x = q[1], y = q[2], z = q[3];
    const float n = std::sqrt(w * w + x * x + y * y + z * z);
    if(n < 1e-8f)
    {
        r[0] = 1.0f; r[1] = 0.0f; r[2] = 0.0f;
        r[3] = 0.0f; r[4] = 1.0f; r[5] = 0.0f;
        r[6] = 0.0f; r[7] = 0.0f; r[8] = 1.0f;
        return;
    }
    w /= n; x /= n; y /= n; z /= n;
    r[0] = 1.0f - 2.0f * (y * y + z * z);
    r[1] = 2.0f * (x * y - w * z);
    r[2] = 2.0f * (x * z + w * y);
    r[3] = 2.0f * (x * y + w * z);
    r[4] = 1.0f - 2.0f * (x * x + z * z);
    r[5] = 2.0f * (y * z - w * x);
    r[6] = 2.0f * (x * z - w * y);
    r[7] = 2.0f * (y * z + w * x);
    r[8] = 1.0f - 2.0f * (x * x + y * y);
}

void transform_point(const float p[3], const float r[9],
                     const float t[3], float out[3])
{
    out[0] = r[0] * p[0] + r[1] * p[1] + r[2] * p[2] + t[0];
    out[1] = r[3] * p[0] + r[4] * p[1] + r[5] * p[2] + t[1];
    out[2] = r[6] * p[0] + r[7] * p[1] + r[8] * p[2] + t[2];
}

void draw_die(const Frame &frame)
{
    if(!frame.has_die)
        return;

    const float hx = frame.die_half[0];
    const float hy = frame.die_half[1];
    const float hz = frame.die_half[2];
    const float local[8][3] = {
        {-hx, -hy, -hz}, { hx, -hy, -hz},
        {-hx,  hy, -hz}, { hx,  hy, -hz},
        {-hx, -hy,  hz}, { hx, -hy,  hz},
        {-hx,  hy,  hz}, { hx,  hy,  hz},
    };
    const int edges[12][2] = {
        {0, 1}, {1, 3}, {3, 2}, {2, 0},
        {4, 5}, {5, 7}, {7, 6}, {6, 4},
        {0, 4}, {1, 5}, {2, 6}, {3, 7},
    };

    float r[9];
    float world[8][3];
    quat_to_matrix(frame.die_quat, r);
    for(int i = 0; i < 8; ++i)
        transform_point(local[i], r, frame.die_pos, world[i]);

    glLineWidth(2.4f);
    glColor4f(1.0f, 0.95f, 0.72f, 1.0f);
    glBegin(GL_LINES);
    for(const auto &edge : edges)
    {
        glVertex3fv(world[edge[0]]);
        glVertex3fv(world[edge[1]]);
    }
    glEnd();
}

void draw_particles(const Frame &frame)
{
    const int fluid_stride = stride_for(frame.n_fluid, g_max_particles);

    glPointSize(g_point_size);
    glBegin(GL_POINTS);
    for(int i = 0; i < frame.n_fluid; i += fluid_stride)
    {
        color_for_density(frame.density[static_cast<std::size_t>(i)]);
        glVertex3f(frame.x[static_cast<std::size_t>(i)],
                   frame.y[static_cast<std::size_t>(i)],
                   frame.z[static_cast<std::size_t>(i)]);
    }
    glEnd();

    if(!g_show_ghosts || frame.n_total <= frame.n_fluid)
        return;

    const int n_ghost = frame.n_total - frame.n_fluid;
    const int ghost_stride = stride_for(n_ghost, g_max_ghosts);
    glPointSize(std::max(2.0f, g_point_size * 0.55f));
    glColor4f(1.0f, 0.18f, 0.18f, 0.30f);
    glBegin(GL_POINTS);
    for(int i = frame.n_fluid; i < frame.n_total; i += ghost_stride)
    {
        glVertex3f(frame.x[static_cast<std::size_t>(i)],
                   frame.y[static_cast<std::size_t>(i)],
                   frame.z[static_cast<std::size_t>(i)]);
    }
    glEnd();
}

void update_title(const Frame *frame)
{
    static auto last = std::chrono::steady_clock::now();
    const auto now = std::chrono::steady_clock::now();
    const double dt =
        std::chrono::duration<double>(now - last).count();
    if(dt < 0.25)
        return;
    last = now;

    char title[256];
    if(frame)
    {
        std::snprintf(title, sizeof(title),
                      "Rigging Magic native viewer | step %d | sim %.3fs | %.1f fps | %s ghosts",
                      frame->step, frame->t, g_render_fps,
                      g_show_ghosts ? "showing" : "hiding");
    }
    else
    {
        std::snprintf(title, sizeof(title),
                      "Rigging Magic native viewer | waiting for stream");
    }
    glutSetWindowTitle(title);
}

void draw_text_2d(float x, float y, const char *text)
{
    glRasterPos2f(x, y);
    for(const char *p = text; *p; ++p)
        glutBitmapCharacter(GLUT_BITMAP_9_BY_15, *p);
}

void draw_overlay(const Frame *frame)
{
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0.0, static_cast<double>(g_width), 0.0,
               static_cast<double>(g_height));

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glDisable(GL_DEPTH_TEST);
    glColor4f(0.94f, 0.96f, 0.98f, 1.0f);

    char line[256];
    if(frame)
    {
        std::snprintf(line,
                      sizeof(line),
                      "render %.1f fps   stream %.1f fps   step %d   sim %.3fs",
                      g_render_fps,
                      g_stream_fps,
                      frame->step,
                      frame->t);
    }
    else
    {
        std::snprintf(line,
                      sizeof(line),
                      "render %.1f fps   stream %.1f fps   waiting for stream",
                      g_render_fps,
                      g_stream_fps);
    }
    draw_text_2d(12.0f, static_cast<float>(g_height - 24), line);

    if(frame)
    {
        double stream_lag_ms = 0.0;
        const uint64_t last_stream_ns = g_last_stream_ns.load();
        if(last_stream_ns != 0)
        {
            stream_lag_ms =
                static_cast<double>(steady_now_ns() - last_stream_ns) / 1.0e6;
        }
        std::snprintf(line,
                      sizeof(line),
                      "particles %d fluid / %d total   stream lag %.0f ms   ghosts %s",
                      frame->n_fluid,
                      frame->n_total,
                      stream_lag_ms,
                      g_show_ghosts ? "on" : "off");
        draw_text_2d(12.0f, static_cast<float>(g_height - 44), line);
    }

    glEnable(GL_DEPTH_TEST);
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
}

void display()
{
    static auto fps_t0 = std::chrono::steady_clock::now();
    static int  frames = 0;
    static uint64_t stream_frames0 = 0;

    std::shared_ptr<Frame> frame;
    std::string error;
    bool done = false;
    {
        std::lock_guard<std::mutex> lock(g_frame_mutex);
        frame = g_latest_frame;
        error = g_stream_error;
        done = g_stream_done.load();
    }

    if(!error.empty())
    {
        std::fprintf(stderr, "[fast_viewer] stream error: %s\n",
                     error.c_str());
        std::exit(1);
    }
    if(done && !frame)
    {
        std::fprintf(stderr, "[fast_viewer] stream ended before first frame\n");
        std::exit(1);
    }

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0f, 0.0f, -g_distance);
    glRotatef(g_pitch, 1.0f, 0.0f, 0.0f);
    glRotatef(g_yaw, 0.0f, 0.0f, 1.0f);

    draw_sphere();
    if(frame)
    {
        draw_particles(*frame);
        draw_die(*frame);
    }

    ++frames;
    const auto now = std::chrono::steady_clock::now();
    const double dt = std::chrono::duration<double>(now - fps_t0).count();
    if(dt >= 1.0)
    {
        g_render_fps = static_cast<double>(frames) / dt;
        const uint64_t stream_frames_now = g_stream_frames.load();
        g_stream_fps =
            static_cast<double>(stream_frames_now - stream_frames0) / dt;
        stream_frames0 = stream_frames_now;
        frames = 0;
        fps_t0 = now;
    }
    draw_overlay(frame.get());
    glutSwapBuffers();
    update_title(frame.get());
}

void reshape(int w, int h)
{
    g_width = std::max(1, w);
    g_height = std::max(1, h);
    glViewport(0, 0, g_width, g_height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0, static_cast<double>(g_width) /
                             static_cast<double>(g_height),
                   0.01, 10.0);
    glMatrixMode(GL_MODELVIEW);
}

void timer(int)
{
    glutPostRedisplay();
    const int ms = std::max(1, static_cast<int>(1000.0f / g_target_fps));
    glutTimerFunc(ms, timer, 0);
}

void keyboard(unsigned char key, int, int)
{
    switch(key)
    {
        case 27:
        case 'q':
            std::exit(0);
            break;
        case 'g':
            g_show_ghosts = !g_show_ghosts;
            break;
        case '+':
        case '=':
            g_point_size = std::min(20.0f, g_point_size + 1.0f);
            break;
        case '-':
        case '_':
            g_point_size = std::max(1.0f, g_point_size - 1.0f);
            break;
        case 'r':
            g_yaw = 35.0f;
            g_pitch = 18.0f;
            g_distance = 0.48f;
            break;
        default:
            break;
    }
}

void mouse(int button, int state, int x, int y)
{
    if(button == GLUT_LEFT_BUTTON)
    {
        g_dragging = state == GLUT_DOWN;
        g_last_mouse_x = x;
        g_last_mouse_y = y;
    }
    else if(state == GLUT_DOWN && button == 3)
    {
        g_distance = std::max(0.08f, g_distance * 0.92f);
    }
    else if(state == GLUT_DOWN && button == 4)
    {
        g_distance = std::min(2.0f, g_distance * 1.08f);
    }
}

void motion(int x, int y)
{
    if(!g_dragging)
        return;
    const int dx = x - g_last_mouse_x;
    const int dy = y - g_last_mouse_y;
    g_last_mouse_x = x;
    g_last_mouse_y = y;
    g_yaw += 0.45f * static_cast<float>(dx);
    g_pitch += 0.45f * static_cast<float>(dy);
    g_pitch = std::max(-85.0f, std::min(85.0f, g_pitch));
}

void init_gl()
{
    glClearColor(0.03f, 0.035f, 0.04f, 1.0f);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_POINT_SMOOTH);
    glHint(GL_POINT_SMOOTH_HINT, GL_FASTEST);
}

void usage(const char *argv0)
{
    std::fprintf(stderr,
                 "usage: %s [-] [--fps N] [--point-size N] "
                 "[--max-particles N] [--max-ghosts N] "
                 "[--ghosts|--no-ghosts]\n",
                 argv0);
}

void parse_args(int argc, char **argv)
{
    for(int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if(arg == "-")
        {
            continue;
        }
        if(arg == "--fps" && i + 1 < argc)
        {
            g_target_fps = std::max(1.0f,
                                    static_cast<float>(std::atof(argv[++i])));
        }
        else if(arg == "--point-size" && i + 1 < argc)
        {
            g_point_size = std::max(1.0f,
                                    static_cast<float>(std::atof(argv[++i])));
        }
        else if(arg == "--max-particles" && i + 1 < argc)
        {
            g_max_particles = std::max(0, std::atoi(argv[++i]));
        }
        else if(arg == "--max-ghosts" && i + 1 < argc)
        {
            g_max_ghosts = std::max(0, std::atoi(argv[++i]));
        }
        else if(arg == "--ghosts")
        {
            g_show_ghosts = true;
        }
        else if(arg == "--no-ghosts")
        {
            g_show_ghosts = false;
        }
        else
        {
            usage(argv[0]);
            std::exit(2);
        }
    }
}

} // namespace

int main(int argc, char **argv)
{
    parse_args(argc, argv);

    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    std::thread(reader_loop).detach();

    char app_name[] = "rigging_magic_fast_viewer";
    char *glut_argv[] = {app_name, nullptr};
    int glut_argc = 1;
    glutInit(&glut_argc, glut_argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
    glutInitWindowSize(g_width, g_height);
    glutCreateWindow("Rigging Magic native viewer | waiting for stream");

    init_gl();
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    timer(0);
    glutMainLoop();
    return 0;
}
