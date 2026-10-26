
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#define M_PI 3.14159265358979323846
#define NAV_IDLE 0
#define NAV_TRAJ 6
#define NAV_ARRIVED 7
#define MISSION_FLYING 1
#define PX4_MAIN_MODE_OFFBOARD 6
#define portMAX_DELAY 0
#define xSemaphoreTake(...) ((void)0)
#define xSemaphoreGive(...) ((void)0)
#define ESP_LOGI(...) ((void)0)
#define ESP_LOGW(...) ((void)0)
typedef struct { float x,y,z; } traj_pt_t;
typedef struct { int state; bool has_goal; } nav_t;
typedef struct { bool armed; int custom_main_mode; float x,y,z,heading; } drone_state_t;
static traj_pt_t points[2], *s_traj=points;
static uint16_t s_traj_n=2,s_traj_dt_ms;
static uint8_t s_traj_id;
static float s_traj_org[3],s_traj_yaw,last_sent_yaw;
static bool s_traj_yaw_follow,valid_position=true;
static int64_t s_traj_t0_us,now_us;
static nav_t s_nav,s_status;
static int s_phase=MISSION_FLYING;
static drone_state_t drone;
static int64_t esp_timer_get_time(void) { return now_us; }
static drone_state_t mavlink_get_state(void) { return drone; }
static bool mavlink_position_valid(void) { return valid_position; }
static void mavlink_set_position_ned(float x,float y,float z,float yaw) { last_sent_yaw=yaw; }
static void mavlink_set_position_velocity_ned(float x,float y,float z,float vx,float vy,float vz,float yaw) { last_sent_yaw=yaw; }

#define TRAJ_YAW_RATE_MAX 1.0f         /* rad/s, yaw-follow turn-rate limit                         */
#define NAV_TICK_MS 50                /* nav_task period; also sets the per-tick yaw limit         */
static void traj_tick(void)
{
    float k    = (float)(esp_timer_get_time() - s_traj_t0_us) / (1000.0f * s_traj_dt_ms);
    int   last = s_traj_n - 1;

    if (k >= (float)last) {
        const traj_pt_t *p = &s_traj[last];
        mavlink_set_position_ned(s_traj_org[0] + p->x, s_traj_org[1] + p->y,
                                 s_traj_org[2] + p->z, s_traj_yaw);
        xSemaphoreTake(s_mutex, portMAX_DELAY);
        if (s_nav.state == NAV_TRAJ) {
            s_nav.state    = NAV_ARRIVED;
            s_nav.has_goal = false;
            s_status.state = NAV_ARRIVED;
        }
        xSemaphoreGive(s_mutex);
        ESP_LOGI(TAG, "Trajectory done — holding last point");
        return;
    }

    int   i  = (int)k;
    float a  = k - (float)i;
    float hz = 1000.0f / (float)s_traj_dt_ms;
    const traj_pt_t *p0 = &s_traj[i], *p1 = &s_traj[i + 1];

    if (s_traj_yaw_follow) {
        /* Heading tangent to the path: NED heading = atan2(east, north) of this segment.
         * Hovers (no motion) keep the last heading; rate limited per navigation tick. */
        float dx = p1->x - p0->x, dy = p1->y - p0->y;
        if (dx * dx + dy * dy > 1e-8f) {
            float want = atan2f(dy, dx);
            float d = want - s_traj_yaw;
            while (d >  (float)M_PI) d -= 2.0f * (float)M_PI;
            while (d < -(float)M_PI) d += 2.0f * (float)M_PI;
            float lim = TRAJ_YAW_RATE_MAX * (NAV_TICK_MS / 1000.0f);
            if (d >  lim) d =  lim;
            if (d < -lim) d = -lim;
            s_traj_yaw += d;
            if (s_traj_yaw >  (float)M_PI) s_traj_yaw -= 2.0f * (float)M_PI;
            if (s_traj_yaw < -(float)M_PI) s_traj_yaw += 2.0f * (float)M_PI;
        }
    }

    mavlink_set_position_velocity_ned(
        s_traj_org[0] + p0->x + a * (p1->x - p0->x),
        s_traj_org[1] + p0->y + a * (p1->y - p0->y),
        s_traj_org[2] + p0->z + a * (p1->z - p0->z),
        (p1->x - p0->x) * hz, (p1->y - p0->y) * hz, (p1->z - p0->z) * hz,
        s_traj_yaw);
}
void nav_traj_start(uint8_t id, uint16_t dt_ms, uint8_t flags)
{
    drone_state_t st = mavlink_get_state();
    bool flying = st.armed && st.custom_main_mode == PX4_MAIN_MODE_OFFBOARD
                  && mavlink_position_valid();
    const char *err = NULL;

    xSemaphoreTake(s_mutex, portMAX_DELAY);
    if (s_nav.state == NAV_TRAJ)          err = "already playing";
    else if (id == 0 || id != s_traj_id) err = "unknown upload id";
    else if (dt_ms == 0)                 err = "dt_ms is 0";
    else if (!flying)                    err = "not armed in OFFBOARD";
    for (int i = 0; !err && i < s_traj_n; i++) {
        if (isnan(s_traj[i].x)) err = "upload incomplete";
    }
    if (!err) {
        s_traj_org[0] = st.x;
        s_traj_org[1] = st.y;
        s_traj_org[2] = st.z;
        s_traj_yaw    = st.heading;
        s_traj_yaw_follow = (flags & 0x01) != 0;
        s_traj_dt_ms  = dt_ms;
        s_traj_t0_us  = esp_timer_get_time();
        s_traj_id     = 0;              /* consume: a repeated START cannot replay */
        s_nav.state    = NAV_TRAJ;
        s_nav.has_goal = true;
        s_status.state = NAV_TRAJ;
    }
    uint16_t n = s_traj_n;
    xSemaphoreGive(s_mutex);

    if (err) {
        ESP_LOGW(TAG, "Trajectory start refused: %s", err);
    } else {
        ESP_LOGI(TAG, "Trajectory start: %u pts @ %u ms from odom (%.2f, %.2f, %.2f)%s",
                 n, dt_ms, st.x, st.y, st.z, (flags & 0x01) ? ", yaw follows path" : "");
    }
}
static void handle_traj_start(const uint8_t *buf, int len)
{
    if (len < 5) return;
    if (s_phase != MISSION_FLYING) {
        ESP_LOGW(TAG, "CMD_TRAJ_START ignored — not flying");
        return;
    }
    uint16_t dt_ms;
    memcpy(&dt_ms, buf + 3, sizeof(dt_ms));
    uint8_t flags = (len >= 6) ? buf[5] : 0;     /* optional: bit0 = yaw follows path */
    nav_traj_start(buf[2], dt_ms, flags);
}

static void reset(float yaw,float dx,float dy) {
 points[0]=(traj_pt_t){0,0,0};points[1]=(traj_pt_t){dx,dy,0};
 s_nav=(nav_t){NAV_IDLE,false};s_status=s_nav;s_traj_id=7;s_traj_yaw_follow=false;
 drone=(drone_state_t){true,PX4_MAIN_MODE_OFFBOARD,0,0,-0.5f,yaw};
 now_us=0;s_phase=MISSION_FLYING;valid_position=true;
}
static void close_to(float actual,float expected) { assert(fabsf(actual-expected)<1e-5f); }
int main(void) {
 uint8_t legacy[]={2,9,7,0xe8,3};
 uint8_t follow[]={2,9,7,0xe8,3,1};
 reset(.4f,0,1);handle_traj_start(legacy,5);assert(!s_traj_yaw_follow);now_us=50000;traj_tick();close_to(last_sent_yaw,.4f);
 reset(0,0,1);handle_traj_start(follow,6);assert(s_traj_yaw_follow);now_us=50000;traj_tick();close_to(last_sent_yaw,.05f);
 for(int i=2;i<=19;i++){ now_us=i*50000;traj_tick(); } close_to(last_sent_yaw,.95f);
 reset(0,0,-1);handle_traj_start(follow,6);now_us=50000;traj_tick();close_to(last_sent_yaw,-.05f);
 reset(.7f,0,0);handle_traj_start(follow,6);now_us=50000;traj_tick();close_to(last_sent_yaw,.7f);
 float target=-(float)M_PI+.02f;
 reset((float)M_PI-.01f,cosf(target),sinf(target));handle_traj_start(follow,6);now_us=50000;traj_tick();close_to(last_sent_yaw,target);
 reset(0,1,0);handle_traj_start(follow,4);assert(s_nav.state==NAV_IDLE);
 reset(0,1,0);s_phase=0;handle_traj_start(follow,6);assert(s_nav.state==NAV_IDLE);
 reset(0,1,0);drone.armed=false;handle_traj_start(follow,6);assert(s_nav.state==NAV_IDLE);
 reset(0,1,0);handle_traj_start(follow,6);now_us=1000000;traj_tick();assert(s_nav.state==NAV_ARRIVED);assert(!s_nav.has_goal);
 puts("PASS: legacy packet, yaw flag, +/- 1 rad/s limit at 50 ms, hover, angle wrap, truncated packet, flight/arming gates, trajectory completion");
 return 0;
}
