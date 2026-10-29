// CUDA Moba — standalone env source for --cu builds. Prototype port of moba.h:
// one thread per env, game logic kept serial and in the same order as the CPU
// version. Differences from moba.h:
//   - per-env xorshift RNG instead of the process-wide rand()
//   - skill function pointers replaced by a switch on hero type
//   - scanned target lists store entity indices instead of Entity*
//   - out-of-map reads/writes are guarded (UB on CPU, a fault on GPU)
//   - debug printf/exit paths dropped; rendering is a minimal map view
#ifndef PUFFER_MOBA_GPU_CU
#define PUFFER_MOBA_GPU_CU

#define PUF_BACKEND PUF_GPU

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "game_map.h"
typedef __nv_bfloat16 obs_t;
#include "pufferenv.h"

#define ACT_SIZES {7, 7, 3, 2, 2, 2}
#define OBS_SIZE 510
#define NUM_ATNS 6
#define MAX_MOBA_AGENTS 10

#define NUM_PLAYERS 10
#define NUM_CREEPS 100
#define NUM_NEUTRALS 72
#define NUM_TOWERS 24
#define NUM_ENTITIES (NUM_PLAYERS + NUM_CREEPS + NUM_NEUTRALS + NUM_TOWERS)
#define CREEP_OFFSET NUM_PLAYERS
#define NEUTRAL_OFFSET (NUM_PLAYERS + NUM_CREEPS)
#define TOWER_OFFSET (NUM_PLAYERS + NUM_CREEPS + NUM_NEUTRALS)

#define EMPTY 0
#define WALL 1
#define TOWER 2
#define RADIANT_CREEP 3
#define DIRE_CREEP 4
#define NEUTRAL 5
#define RADIANT_SUPPORT 6
#define RADIANT_ASSASSIN 7
#define RADIANT_BURST 8
#define RADIANT_TANK 9
#define RADIANT_CARRY 10

#define TOWER_VISION 5
#define CREEP_VISION 7
#define NEUTRAL_VISION 3

#define ENTITY_PLAYER 0
#define ENTITY_CREEP 1
#define ENTITY_NEUTRAL 2
#define ENTITY_TOWER 3

#define XP_RANGE 7
#define MAP_N 128
#define MAP_CELLS (MAP_N * MAP_N)
// Same flat layout as moba.h's Entity* scanned_targets[256][121]: lists that
// overrun 121 entries spill into the next row exactly as they do on CPU.
#define SCAN_ROWS 256
#define SCAN_COLS 121
#define MOBA_BLOCK 64

struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float reward_death;
    float reward_xp;
    float reward_distance;
    float reward_tower;

    float radiant_victory;
    float radiant_level;
    float radiant_towers_alive;

    float dire_victory;
    float dire_level;
    float dire_towers_alive;

    float radiant_support_episode_return;
    float radiant_support_reward_death;
    float radiant_support_reward_xp;
    float radiant_support_reward_distance;
    float radiant_support_reward_tower;
    float radiant_support_level;
    float radiant_support_kills;
    float radiant_support_deaths;
    float radiant_support_damage_dealt;
    float radiant_support_damage_received;
    float radiant_support_healing_dealt;
    float radiant_support_healing_received;
    float radiant_support_creeps_killed;
    float radiant_support_neutrals_killed;
    float radiant_support_towers_killed;
    float radiant_support_usage_auto;
    float radiant_support_usage_q;
    float radiant_support_usage_w;
    float radiant_support_usage_e;
    float n;
};

void puf_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "radiant_victory", log->radiant_victory);
    dict_set(out, "dire_victory", log->dire_victory);
    dict_set(out, "radiant_level", log->radiant_level);
    dict_set(out, "dire_level", log->dire_level);
    dict_set(out, "radiant_towers_alive", log->radiant_towers_alive);
    dict_set(out, "dire_towers_alive", log->dire_towers_alive);
    dict_set(out, "n", log->n);
}

typedef struct PlayerLog {
    float episode_return;
    float reward_death;
    float reward_xp;
    float reward_distance;
    float reward_tower;
    float level;
    float kills;
    float deaths;
    float damage_dealt;
    float damage_received;
    float healing_dealt;
    float healing_received;
    float creeps_killed;
    float neutrals_killed;
    float towers_killed;
    float usage_auto;
    float usage_q;
    float usage_w;
    float usage_e;
} PlayerLog;

typedef struct Entity {
    int pid;
    int entity_type;
    int hero_type;
    int grid_id;
    int team;
    float health;
    float max_health;
    float mana;
    float max_mana;
    float y;
    float x;
    float spawn_y;
    float spawn_x;
    float damage;
    int lane;
    int waypoint;
    float move_speed;
    float move_modifier;
    int stun_timer;
    int move_timer;
    int q_timer;
    int w_timer;
    int e_timer;
    int basic_attack_timer;
    int basic_attack_cd;
    int is_hit;
    int level;
    int xp;
    int xp_on_kill;
    float reward;
    int tier;
    float base_health;
    float base_mana;
    float base_damage;
    int hp_gain_per_level;
    int mana_gain_per_level;
    int damage_gain_per_level;
    float last_x;
    float last_y;
    int target_pid;
    int attack_aoe;
} Entity;

typedef struct Reward {
    float death;
    float xp;
    float distance;
    float tower;
} Reward;

// Per-env state (~140 KB). Trainer-facing fields match the CPU Env contract.
typedef struct Env {
    Log log;
    Agent agents[MAX_MOBA_AGENTS];
    int num_agents;
    int tag;
    int boundary_reached;
    int tick;
    int creep_idx;
    int total_towers_taken;
    int total_levels_gained;
    unsigned int rng;
    Entity entities[NUM_ENTITIES];
    Reward reward_components[NUM_PLAYERS];
    PlayerLog player_logs[NUM_PLAYERS];
    unsigned char grid[MAP_CELLS];
    short pids[MAP_CELLS];
    short scanned[SCAN_ROWS * SCAN_COLS];
} Env;

typedef struct MobaConfig {
    int vision_range;
    float agent_speed;
    int script_opponents;
    int num_agents;
    float reward_death;
    float reward_xp;
    float reward_distance;
    float reward_tower;
    const unsigned char* paths;     // shared 128^4 next-step table
    const unsigned char* orig_grid; // static map
} MobaConfig;

__constant__ MobaConfig d_mcfg;

__constant__ float XP_FOR_LEVEL[] = {0, 240, 640, 1160, 1760, 2440, 3200, 4000, 4900, 4900, 7000, 8200, 9500, 10900, 12400, 14000, 15700, 17500, 19400, 21400, 23600, 26000, 28600, 31400, 34400, 38400, 43400, 49400, 56400, 63900};

__constant__ float ATN_MAP[][8] = {
    {1, -1, 0, 0, 1, -1, -1, 1},
    {0, 0, 1, -1, -1, -1, 1, 1}
};

__constant__ float NEUTRAL_CAMP_X[] = {44.753846153846155, 74.41538461538462, 101.67692307692307, 89.92307692307692, 73.95384615384616, 64.38461538461539, 31.657692307692308, 95.67692307692307, 81.1076923076923, 34.99230769230769, 50.784615384615385, 63.646153846153844, 59.49230769230769, 44.69230769230769, 98.67307692307692, 28.642307692307693, 64.87692307692308, 51.46153846153846};
__constant__ float NEUTRAL_CAMP_Y[] = {71.99230769230769, 108.15384615384616, 102.16923076923078, 102.78461538461539, 40.753846153846155, 39.92307692307692, 39.96923076923077, 70.78461538461538, 69.18461538461538, 59.52307692307692, 99.95384615384614, 93.97692307692307, 49.86153846153846, 31.353846153846156, 61.06153846153846, 69.92307692307692, 83.83076923076923, 33.98461538461538};
__constant__ float TOWER_DAMAGE[] = {175.0, 175.0, 175.0, 175.0, 110.0, 175.0, 175.0, 110.0, 175.0, 175.0, 175.0, 110.0, 175.0, 110.0, 175.0, 175.0, 175.0, 175.0, 175.0, 175.0, 110.0, 110.0, 0, 0};
__constant__ float TOWER_HEALTH[] = {2000, 2000, 2000, 2000, 1800, 2000, 2000, 1800, 2100, 2100, 2000, 1800, 2000, 1800, 2000, 2000, 2000, 2000, 2100, 2100, 1800, 1800, 4500, 4500};
__constant__ int TOWER_TEAM[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0};
__constant__ int TOWER_TIER[] = {3, 3, 3, 2, 1, 2, 2, 1, 4, 4, 2, 1, 2, 1, 2, 3, 3, 3, 4, 4, 1, 1, 5, 5};
__constant__ float TOWER_X[] = {34.6, 29.307692307692307, 14.292307692307695, 64.2, 102.87692307692308, 38.29230769230769, 17.615384615384613, 16.876923076923077, 21.06153846153846, 103.03076923076924, 65.0, 29.06153846153846, 84.2, 69.03076923076924, 112.75384615384615, 113.73846153846154, 92.32307692307693, 97.86153846153846, 105.61538461538461, 23.52307692307692, 53.12307692307692, 113.22307692307692, 107.52307692307693, 19.46153846153846};
__constant__ float TOWER_Y[] = {112.01538461538462, 96.87692307692308, 91.21538461538461, 113.61538461538461, 112.13846153846154, 85.43076923076923, 71.70769230769231, 51.03076923076923, 102.41538461538462, 28.261538461538464, 18.723076923076924, 18.723076923076924, 48.753846153846155, 59.98461538461538, 62.04615384615384, 41.676923076923075, 20.56923076923077, 36.08461538461539, 30.907692307692308, 104.93846153846154, 75.83076923076923, 78.3, 26.53846153846154, 106.16923076923078};
__constant__ float WAYPOINTS[][20][2] = {{{96.26153846153846, 14.04615384615385}, {93.61538461538461, 14.292307692307695}, {58.23076923076923, 16.07692307692308}, {43.4, 16.01538461538462}, {32.10769230769231, 17.584615384615383}, {24.93846153846154, 19.123076923076923}, {22.96923076923077, 20.446153846153848}, {21.06153846153846, 25.307692307692307}, {20.200000000000003, 30.47692307692308}, {19.4, 35.892307692307696}, {18.47692307692308, 43.52307692307693}, {18.04615384615385, 50.723076923076924}, {20.50769230769231, 94.66153846153847}, {27.676923076923075, 105.94615384615385}}, {{99.52307692307693, 26.47692307692308}, {98.66153846153847, 27.646153846153844}, {86.04615384615384, 41.12307692307692}, {71.33846153846154, 57.49230769230769}, {66.96923076923076, 62.23076923076923}, {64.07692307692308, 66.35384615384615}, {60.2, 72.01538461538462}, {51.892307692307696, 84.87692307692308}, {34.66153846153846, 99.4}, {28.169230769230772, 105.94615384615385}}, {{112.01538461538462, 36.93846153846154}, {112.01538461538462, 32.50769230769231}, {116.2, 68.6923076923077}, {113.73846153846154, 80.75384615384615}, {113.55384615384615, 90.53846153846155}, {113.21538461538461, 95.98461538461538}, {111.49230769230769, 100.93846153846154}, {109.06153846153846, 108.07692307692308}, {107.18461538461538, 111.36923076923077}, {101.73846153846154, 114.66153846153847}, {90.53846153846155, 111.76923076923077}, {39.4, 113.73846153846154}, {28.169230769230772, 106.43846153846154}}, {{20.446153846153848, 89.36923076923077}, {20.630769230769232, 94.66153846153847}, {18.784615384615385, 50.6}, {22.292307692307695, 24.50769230769231}, {22.815384615384616, 18.815384615384616}, {27.52307692307692, 17.03076923076923}, {35.03076923076923, 15.95384615384615}, {93.61538461538461, 14.415384615384617}, {103.64615384615385, 21.99230769230769}}, {{37.43076923076923, 96.50769230769231}, {34.53846153846154, 99.27692307692308}, {51.707692307692305, 84.38461538461539}, {59.83076923076923, 71.64615384615385}, {63.83076923076923, 66.1076923076923}, {66.72307692307692, 61.98461538461538}, {70.84615384615384, 57.246153846153845}, {98.53846153846155, 27.52307692307692}, {103.64615384615385, 22.48461538461538}}, {{36.93846153846154, 113.24615384615385}, {39.4, 113.61538461538461}, {62.292307692307695, 114.6}, {83.64615384615385, 114.6}, {90.66153846153847, 109.8}, {94.87692307692308, 112.01538461538462}, {104.26153846153846, 112.2923076923077}, {109.3076923076923, 107.83076923076922}, {112.23076923076923, 105.86153846153846}, {114.44615384615385, 72.96923076923076}, {111.8923076923077, 32.50769230769231}, {104.13846153846154, 22.48461538461538}}};
__constant__ int WAYPOINTS_N[6] = {14, 10, 13, 9, 9, 12};

// ---- helpers ----

// rand() stand-in: non-negative 31-bit ints from a per-env xorshift32.
__device__ static inline int mb_rand(Env* env) {
    unsigned int x = env->rng;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    env->rng = x;
    return (int)(x >> 1);
}

// fast_rng stand-in: uniform in [-1, 1).
__device__ static inline float mb_jitter(Env* env) {
    return -1.0f + 2.0f * (float)mb_rand(env) * (1.0f / 2147483648.0f);
}

__device__ static inline int mb_in_map(int adr) {
    return adr >= 0 && adr < MAP_CELLS;
}

__device__ static inline int mb_offset(int y, int x) {
    return y*MAP_N + x;
}

__device__ static inline float mb_l1(float x1, float y1, float x2, float y2) {
    return fabsf(x1 - x2) + fabsf(y1 - y2);
}

__device__ static inline short* mb_scan_row(Env* env, int pid) {
    return &env->scanned[pid * SCAN_COLS];
}

__device__ static inline int mb_calc_level(int xp) {
    int i;
    for (i = 0; i < 30; i++) {
        if (xp < XP_FOR_LEVEL[i])
            return i + 1;
    }
    return i;
}

// ---- movement ----

__device__ static int mb_move_to(Env* env, Entity* player, float dest_y, float dest_x) {
    int src = mb_offset((int)player->y, (int)player->x);
    int dst = mb_offset((int)dest_y, (int)dest_x);
    if (!mb_in_map(dst))
        return 1;

    if (env->grid[dst] != EMPTY && env->pids[dst] != player->pid)
        return 1;

    if (mb_in_map(src)) {
        env->grid[src] = EMPTY;
        env->pids[src] = -1;
    }
    env->grid[dst] = player->grid_id;
    env->pids[dst] = player->pid;

    player->y = dest_y;
    player->x = dest_x;
    return 0;
}

__device__ static int mb_move_near(Env* env, Entity* entity, Entity* target) {
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (mb_move_to(env, entity, target->y + dy, target->x + dx) == 0)
                return 0;
        }
    }
    return 1;
}

__device__ static int mb_move_towards(Env* env, Entity* entity, int y_dst, int x_dst, float speed) {
    int y_src = entity->y;
    int x_src = entity->x;
    if ((unsigned)y_dst >= MAP_N || (unsigned)x_dst >= MAP_N
            || (unsigned)y_src >= MAP_N || (unsigned)x_src >= MAP_N)
        return 0;

    size_t adr = (size_t)y_dst*MAP_N*MAP_N*MAP_N + (size_t)x_dst*MAP_N*MAP_N
        + (size_t)y_src*MAP_N + x_src;
    int atn = __ldg(&d_mcfg.paths[adr]);
    // 255 = unreachable; the table is complete, so there is nothing to fill
    if (atn >= 8)
        return 0;

    float modifier = speed * entity->move_modifier;
    y_dst = y_src + modifier*ATN_MAP[0][atn];
    x_dst = x_src + modifier*ATN_MAP[1][atn];

    if (mb_move_to(env, entity, y_dst, x_dst) == 0)
        return 0;

    float jitter_x = mb_jitter(env);
    float jitter_y = mb_jitter(env);
    return mb_move_to(env, entity, entity->y + jitter_y, entity->x + jitter_x);
}

__device__ static void mb_kill_entity(Env* env, Entity* entity) {
    if (entity->pid == -1)
        return;
    int adr = mb_offset((int)entity->y, (int)entity->x);
    if (mb_in_map(adr)) {
        env->grid[adr] = EMPTY;
        env->pids[adr] = -1;
    }
    entity->pid = -1;
    entity->target_pid = -1;
    entity->health = 0;
    entity->last_x = 0;
    entity->last_y = 0;
    entity->x = 0;
    entity->y = 0;
}

__device__ static void mb_spawn_player(Env* env, Entity* entity) {
    int pid = entity->pid;
    mb_kill_entity(env, entity);
    entity->pid = pid;

    entity->max_health = entity->base_health + entity->level*entity->hp_gain_per_level;
    entity->max_mana = entity->base_mana + entity->level*entity->mana_gain_per_level;
    entity->damage = entity->base_damage + entity->level*entity->damage_gain_per_level;

    entity->health = entity->max_health;
    entity->mana = entity->max_mana;
    entity->basic_attack_timer = 0;
    entity->move_modifier = 0;
    entity->move_timer = 0;
    entity->stun_timer = 0;
    entity->q_timer = 0;
    entity->w_timer = 0;
    entity->e_timer = 0;
    entity->target_pid = -1;
    entity->waypoint = 1;

    int y, x;
    while (true) {
        y = entity->spawn_y + mb_rand(env)%15 - 7;
        x = entity->spawn_x + mb_rand(env)%15 - 7;
        int adr = mb_offset(y, x);
        if (mb_in_map(adr) && env->grid[adr] == EMPTY)
            break;
    }
    entity->last_x = x;
    entity->last_y = y;
    mb_move_to(env, entity, y, x);
}

// ---- combat ----

__device__ static int mb_attack(Env* env, Entity* player, Entity* target, float damage) {
    if (target->pid == -1 || target->team == player->team)
        return 1;

    float dist_to_target = mb_l1(player->y, player->x, target->y, target->x);
    if (dist_to_target > 12)
        return 1;

    PlayerLog empty_log = {0};
    PlayerLog* player_log = &empty_log;
    if (player->entity_type == ENTITY_PLAYER)
        player_log = &env->player_logs[player->pid];
    PlayerLog* target_log = &empty_log;
    if (target->entity_type == ENTITY_PLAYER)
        target_log = &env->player_logs[target->pid];

    if (damage < target->health) {
        player_log->damage_dealt += damage;
        target_log->damage_received += damage;
        target->health -= damage;
        player->target_pid = target->pid;
        target->is_hit = 1;
        return 0;
    }

    player_log->damage_dealt += target->health;
    target_log->damage_received += target->health;
    target->health = 0;

    int target_type = target->entity_type;
    if (target_type == ENTITY_PLAYER) {
        env->reward_components[target->pid].death = d_mcfg.reward_death;
        target_log->reward_death = d_mcfg.reward_death;
        player_log->kills += 1;
        target_log->deaths += 1;
        mb_spawn_player(env, target);
    } else if (target_type == ENTITY_CREEP) {
        player_log->creeps_killed += 1;
        mb_kill_entity(env, target);
    } else if (target_type == ENTITY_NEUTRAL) {
        player_log->neutrals_killed += 1;
        mb_kill_entity(env, target);
    } else if (target_type == ENTITY_TOWER) {
        player_log->towers_killed += 1;
        mb_kill_entity(env, target);
    }

    if (player->entity_type != ENTITY_PLAYER)
        return 0;

    int xp = 0;
    if (target_type == ENTITY_PLAYER)
        xp = 100 + (int)(target->xp / 7.69f);
    else
        xp = target->xp_on_kill;

    // Share xp with allies in range
    int first_player_on_team = (player->team == 0) ? 0 : 5;
    bool in_range[5] = {false, false, false, false, false};
    float target_x = target->x;
    float target_y = target->y;
    int num_in_range = 0;
    for (int i = 0; i < 5; i++) {
        Entity* ally = &env->entities[first_player_on_team + i];
        if (ally->pid == player->pid) {
            in_range[i] = true;
            num_in_range += 1;
            continue;
        }
        if (mb_l1(ally->y, ally->x, target_y, target_x) <= XP_RANGE) {
            in_range[i] = true;
            num_in_range += 1;
        }
    }

    xp /= num_in_range;

    for (int i = 0; i < 5; i++) {
        if (!in_range[i])
            continue;

        Entity* ally = &env->entities[first_player_on_team + i];
        if (ally->xp > 10000000)
            continue;

        ally->xp += xp;
        env->reward_components[first_player_on_team + i].xp = xp*d_mcfg.reward_xp;

        int level = ally->level;
        ally->level = mb_calc_level(ally->xp);
        if (ally->level > level) {
            env->player_logs[ally->pid].level = ally->level;
            env->total_levels_gained += 1;
        }

        ally->max_health = ally->base_health + ally->level*ally->hp_gain_per_level;
        ally->max_mana = ally->base_mana + ally->level*ally->mana_gain_per_level;
        ally->damage = ally->base_damage + ally->level*ally->damage_gain_per_level;
    }

    if (target->entity_type == ENTITY_TOWER) {
        env->reward_components[player->pid].tower = d_mcfg.reward_tower;
        env->total_towers_taken += 1;
    }
    return 0;
}

__device__ static int mb_basic_attack(Env* env, Entity* player, Entity* target) {
    if (player->basic_attack_timer > 0)
        return 1;

    player->basic_attack_timer = player->basic_attack_cd;
    return mb_attack(env, player, target, player->damage);
}

__device__ static int mb_heal(Env* env, Entity* player, Entity* target, float amount) {
    if (target->pid == -1 || target->team != player->team)
        return 1;

    // Currently only allowed to heal players
    if (target->entity_type != ENTITY_PLAYER)
        return 1;

    PlayerLog* player_log = &env->player_logs[player->pid];
    PlayerLog* target_log = &env->player_logs[target->pid];
    int missing_health = target->max_health - target->health;
    if (amount <= missing_health) {
        target->health += amount;
        player_log->healing_dealt += amount;
        target_log->healing_received += amount;
        return 0;
    }

    target->health = target->max_health;
    player_log->healing_dealt += missing_health;
    target_log->healing_received += missing_health;
    return 0;
}

__device__ static int mb_spawn_creep(Env* env, int idx, int lane) {
    int pid = CREEP_OFFSET + idx;
    Entity* creep = &env->entities[pid];

    if (lane < 3) {
        creep->team = 0;
        creep->grid_id = RADIANT_CREEP;
    } else {
        creep->team = 1;
        creep->grid_id = DIRE_CREEP;
    }
    creep->pid = pid;
    creep->entity_type = ENTITY_CREEP;
    creep->health = 450;
    creep->max_health = 450;
    creep->lane = lane;
    creep->waypoint = 0;
    creep->xp_on_kill = 60;
    creep->damage = 22;
    creep->basic_attack_cd = 5;

    int spawn_y = WAYPOINTS[lane][0][0];
    int spawn_x = WAYPOINTS[lane][0][1];

    int y, x;
    for (int i = 0; i < 10; i++) {
        y = spawn_y + mb_rand(env) % 7 - 3;
        x = spawn_x + mb_rand(env) % 7 - 3;
        int adr = mb_offset(y, x);
        if (mb_in_map(adr) && env->grid[adr] == EMPTY)
            break;
    }
    creep->health = creep->max_health;
    creep->target_pid = -1;
    creep->waypoint = 1;
    creep->last_x = x;
    creep->last_y = y;
    return mb_move_to(env, creep, y, x);
}

__device__ static int mb_spawn_neutral(Env* env, int idx) {
    int pid = NEUTRAL_OFFSET + idx;
    Entity* neutral = &env->entities[pid];
    neutral->pid = pid;
    neutral->health = neutral->max_health;
    neutral->basic_attack_timer = 0;
    neutral->target_pid = -1;

    int y, x;
    int spawn_y = (int)neutral->spawn_y;
    int spawn_x = (int)neutral->spawn_x;
    for (int i = 0; i < 100; i++) {
        y = spawn_y + mb_rand(env) % 7 - 3;
        x = spawn_x + mb_rand(env) % 7 - 3;
        int adr = mb_offset(y, x);
        if (mb_in_map(adr) && env->grid[adr] == EMPTY)
            break;
    }
    neutral->last_x = x;
    neutral->last_y = y;
    return mb_move_to(env, neutral, y, x);
}

__device__ static int mb_scan_aoe(Env* env, Entity* player, int radius,
        bool exclude_friendly, bool exclude_hostile, bool exclude_creeps,
        bool exclude_neutrals, bool exclude_towers) {
    int player_y = player->y;
    int player_x = player->x;
    int player_team = player->team;
    short* row = mb_scan_row(env, player->pid);
    int idx = 0;

    for (int y = player_y-radius; y <= player_y+radius; y++) {
        for (int x = player_x-radius; x <= player_x+radius; x++) {
            if (y < 0 || y >= MAP_N || x < 0 || x >= MAP_N)
                continue;

            int target_pid = env->pids[mb_offset(y, x)];
            if (target_pid == -1)
                continue;

            Entity* target = &env->entities[target_pid];

            int target_team = target->team;
            if (exclude_friendly && target_team == player_team)
                continue;
            if (exclude_hostile && target_team != player_team)
                continue;

            int target_type = target->entity_type;
            if (exclude_neutrals && target_type == ENTITY_NEUTRAL)
                continue;
            if (exclude_creeps && target_type == ENTITY_CREEP)
                continue;
            if (exclude_towers && target_type == ENTITY_TOWER)
                continue;

            row[idx] = target_pid;
            idx += 1;
        }
    }
    row[idx] = -1;
    return (idx == 0) ? 1 : 0;
}

__device__ static Entity* mb_nearest_scanned_target(Env* env, Entity* player) {
    Entity* nearest_target = NULL;
    float nearest_dist = 9999999;
    float player_y = player->y;
    float player_x = player->x;
    short* row = mb_scan_row(env, player->pid);

    for (int idx = 0; idx < SCAN_COLS; idx++) {
        int tidx = row[idx];
        if (tidx == -1)
            break;

        Entity* target = &env->entities[tidx];
        float dist = mb_l1(player_y, player_x, target->y, target->x);
        if (dist < nearest_dist) {
            nearest_target = target;
            nearest_dist = dist;
        }
    }
    return nearest_target;
}

__device__ static void mb_aoe_scanned(Env* env, Entity* player, float damage, int stun) {
    short* row = mb_scan_row(env, player->pid);
    for (int idx = 0; idx < SCAN_COLS; idx++) {
        int tidx = row[idx];
        if (tidx == -1)
            break;

        Entity* target = &env->entities[tidx];
        // Negative damage is healing
        if (damage < 0) {
            mb_heal(env, player, target, -damage);
            continue;
        }

        mb_attack(env, player, target, damage);
        if (stun > 0)
            target->stun_timer = stun;
    }
}

__device__ static int mb_player_aoe_attack(Env* env, Entity* player,
        Entity* target, int radius, float damage, int stun) {
    bool exclude_hostile = damage < 0;
    bool exclude_friendly = !exclude_hostile;

    int err = mb_scan_aoe(env, player, radius, exclude_friendly,
        exclude_hostile, false, false, false);
    if (err != 0)
        return 1;

    mb_aoe_scanned(env, player, damage, stun);
    player->target_pid = target->pid;
    player->attack_aoe = radius;
    return 0;
}

__device__ static int mb_push(Env* env, Entity* player, Entity* target, float amount) {
    float dx = target->x - player->x;
    float dy = target->y - player->y;
    float dist = fabsf(dx) + fabsf(dy);

    if (dist == 0.0f)
        return 1;

    // Norm to unit vector
    dx = amount * dx / dist;
    dy = amount * dy / dist;
    return mb_move_to(env, target, target->y + dy, target->x + dx);
}

__device__ static int mb_pull(Env* env, Entity* player, Entity* target, float amount) {
    return mb_push(env, player, target, -amount);
}

__device__ static int mb_aoe_pull(Env* env, Entity* player, int radius, float amount) {
    mb_scan_aoe(env, player, radius, true, false, false, false, true);
    int err = 1;
    short* row = mb_scan_row(env, player->pid);
    for (int idx = 0; idx < SCAN_COLS; idx++) {
        int tidx = row[idx];
        if (tidx == -1)
            break;

        mb_pull(env, &env->entities[tidx], player, amount);
        err = 0;
    }
    return err;
}

// ---- AI ----

__device__ static void mb_creep_ai(Env* env, Entity* creep) {
    int waypoint = creep->waypoint;
    int lane = creep->lane;

    if (env->tick % 5 == 0)
        mb_scan_aoe(env, creep, CREEP_VISION, true, false, false, true, false);

    if (mb_scan_row(env, creep->pid)[0] != -1) {
        Entity* target = mb_nearest_scanned_target(env, creep);
        float dest_y = target->y;
        float dest_x = target->x;
        float dist = mb_l1(creep->y, creep->x, dest_y, dest_x);
        if (dist < 2)
            mb_basic_attack(env, creep, target);

        mb_move_towards(env, creep, dest_y, dest_x, d_mcfg.agent_speed);
    } else {
        float dest_y = WAYPOINTS[lane][waypoint][0];
        float dest_x = WAYPOINTS[lane][waypoint][1];
        mb_move_towards(env, creep, dest_y, dest_x, d_mcfg.agent_speed);

        float dist = mb_l1(creep->y, creep->x, dest_y, dest_x);
        if (dist < 2 && creep->waypoint < WAYPOINTS_N[lane] - 1)
            creep->waypoint += 1;
    }
}

__device__ static void mb_neutral_ai(Env* env, Entity* neutral) {
    if (env->tick % 5 == 0)
        mb_scan_aoe(env, neutral, NEUTRAL_VISION, true, false, true, true, true);

    if (mb_scan_row(env, neutral->pid)[0] != -1) {
        Entity* target = mb_nearest_scanned_target(env, neutral);
        if (mb_l1(neutral->y, neutral->x, target->y, target->x) < 2)
            mb_basic_attack(env, neutral, target);
        else
            mb_move_towards(env, neutral, target->y, target->x, d_mcfg.agent_speed);
    } else if (mb_l1(neutral->y, neutral->x,
            neutral->spawn_y, neutral->spawn_x) > 2) {
        mb_move_towards(env, neutral, neutral->spawn_y, neutral->spawn_x, d_mcfg.agent_speed);
    }
}

__device__ static void mb_update_status(Entity* entity) {
    if (entity->stun_timer > 0)
        entity->stun_timer -= 1;
    if (entity->move_timer > 0)
        entity->move_timer -= 1;
    if (entity->move_timer == 0)
        entity->move_modifier = 1.0f;
}

__device__ static void mb_update_cooldowns(Entity* entity) {
    if (entity->q_timer > 0)
        entity->q_timer -= 1;
    if (entity->w_timer > 0)
        entity->w_timer -= 1;
    if (entity->e_timer > 0)
        entity->e_timer -= 1;
    if (entity->basic_attack_timer > 0)
        entity->basic_attack_timer -= 1;
}

// ---- skills (moba.h skill_* in the same order; slot 0/1/2 = q/w/e) ----

__device__ static int mb_skill(Env* env, Entity* player, Entity* target, int slot) {
    int lvl = player->level;
    int mana_cost;
    switch (player->hero_type * 3 + slot) {
    case 0: // support hook
        mana_cost = 100;
        if (target == NULL || player->mana < mana_cost)
            return 1;
        mb_pull(env, target, player, 1.5f + 0.1f*lvl);
        player->mana -= mana_cost;
        player->q_timer = 15;
        return 0;
    case 1: // support aoe heal
        mana_cost = 100;
        if (player->mana < mana_cost)
            return 1;
        if (mb_player_aoe_attack(env, player, player, 5, -350 - 50*lvl, 0) == 0) {
            player->mana -= mana_cost;
            player->w_timer = 50;
            return 0;
        }
        return 1;
    case 2: // support stun
        mana_cost = 75;
        if (target == NULL || player->mana < mana_cost)
            return 1;
        if (mb_attack(env, player, target, 50 + 20*lvl) == 0) {
            target->stun_timer = 15 + 0.5f*lvl;
            player->mana -= mana_cost;
            player->e_timer = 60;
            return 0;
        }
        return 1;
    case 3: { // assassin aoe minions
        mana_cost = 100;
        if (target == NULL || player->mana < mana_cost)
            return 1;
        int target_type = target->entity_type;
        if (target_type != ENTITY_CREEP && target_type != ENTITY_NEUTRAL)
            return 1;
        if (mb_player_aoe_attack(env, player, target, 3, 100 + 20*lvl, 0) == 0) {
            player->mana -= mana_cost;
            player->q_timer = 40;
            return 0;
        }
        return 1;
    }
    case 4: // assassin tp damage
        mana_cost = 150;
        if (target == NULL || player->mana < mana_cost)
            return 1;
        if (mb_move_near(env, player, target) != 0)
            return 1;
        player->mana -= mana_cost;
        if (mb_attack(env, player, target, 250 + 50*lvl) == 0) {
            player->w_timer = 60;
            return 0;
        }
        return 1;
    case 5: // assassin move buff
        mana_cost = 100;
        if (player->mana < mana_cost)
            return 1;
        player->move_modifier = 2.0f;
        player->move_timer = 25;
        player->mana -= mana_cost;
        player->e_timer = 100;
        return 0;
    case 6: // burst nuke
        mana_cost = 200;
        if (target == NULL || player->mana < mana_cost)
            return 1;
        if (mb_attack(env, player, target, 250 + 40*lvl) == 0) {
            player->mana -= mana_cost;
            player->q_timer = 70;
            return 0;
        }
        return 1;
    case 7: // burst aoe
        mana_cost = 100;
        if (target == NULL || player->mana < mana_cost)
            return 1;
        if (mb_player_aoe_attack(env, player, target, 2, 100 + 40*lvl, 0) == 0) {
            player->mana -= mana_cost;
            player->w_timer = 40;
            return 0;
        }
        return 1;
    case 8: // burst aoe stun
        mana_cost = 75;
        if (target == NULL || player->mana < mana_cost)
            return 1;
        if (mb_player_aoe_attack(env, player, target, 2, 0, 10 + 0.5f*lvl) == 0) {
            player->mana -= mana_cost;
            player->e_timer = 50;
            return 0;
        }
        return 1;
    case 9: // tank aoe dot
        mana_cost = 5;
        if (player->mana < mana_cost)
            return 1;
        if (mb_player_aoe_attack(env, player, player, 2, 25 + 2.0f*lvl, 0) == 0) {
            player->mana -= mana_cost;
            return 0;
        }
        return 1;
    case 10: // tank self heal
        mana_cost = 100;
        if (player->mana < mana_cost)
            return 1;
        if (mb_heal(env, player, player, 400 + 125*lvl) == 0) {
            player->mana -= mana_cost;
            player->w_timer = 70;
            return 0;
        }
        return 1;
    case 11: // tank engage aoe
        mana_cost = 50;
        if (target == NULL || player->mana < mana_cost)
            return 1;
        if (mb_move_near(env, player, target) == 0) {
            player->mana -= mana_cost;
            player->e_timer = 40;
            mb_aoe_pull(env, player, 4, 2.0f + 0.1f*lvl);
            return 0;
        }
        return 1;
    case 12: { // carry retreat slow
        mana_cost = 25;
        if (target == NULL || player->mana < mana_cost)
            return 1;
        int err = 1;
        for (int i = 0; i < 3; i++) {
            if (player->mana < mana_cost)
                return err;
            if (mb_push(env, target, player, 3 + 0.1f*lvl) == 0) {
                target->move_timer = 15;
                target->move_modifier = 0.5f;
                player->mana -= mana_cost;
                player->q_timer = 40;
                err = 0;
            }
        }
        return err;
    }
    case 13: // carry slow damage
        mana_cost = 150;
        if (target == NULL || player->mana < mana_cost)
            return 1;
        if (mb_attack(env, player, target, 50 + 20*lvl) == 0) {
            target->move_timer = 20 + lvl;
            target->move_modifier = 0.5f;
            player->mana -= mana_cost;
            player->w_timer = 40;
            return 0;
        }
        return 1;
    case 14: // carry aoe
        mana_cost = 100;
        if (target == NULL || player->mana < mana_cost)
            return 1;
        if (mb_player_aoe_attack(env, player, target, 2, 100 + 20*lvl, 0) == 0) {
            player->mana -= mana_cost;
            player->e_timer = 40;
            return 0;
        }
        return 1;
    }
    return 1;
}

// ---- per-tick systems ----

__device__ static void mb_step_creeps(Env* env) {
    // Spawn wave
    if (env->tick % 150 == 0) {
        for (int lane = 0; lane < 6; lane++) {
            for (int i = 0; i < 5; i++) {
                int creep_pid = CREEP_OFFSET + env->creep_idx;
                mb_kill_entity(env, &env->entities[creep_pid]);
                mb_spawn_creep(env, env->creep_idx, lane);
                env->creep_idx = (env->creep_idx + 1) % NUM_CREEPS;
            }
        }
    }
    for (int idx = 0; idx < NUM_CREEPS; idx++) {
        Entity* creep = &env->entities[CREEP_OFFSET + idx];
        if (creep->pid == -1)
            continue;

        mb_update_status(creep);
        mb_update_cooldowns(creep);
        if (creep->stun_timer > 0)
            continue;

        mb_creep_ai(env, creep);
    }
}

__device__ static void mb_step_neutrals(Env* env) {
    if (env->tick % 600 == 0) {
        for (int camp = 0; camp < 18; camp++) {
            for (int neut = 0; neut < 4; neut++) {
                mb_spawn_neutral(env, 4*camp + neut);
            }
        }
    }
    for (int idx = 0; idx < NUM_NEUTRALS; idx++) {
        Entity* neutral = &env->entities[NEUTRAL_OFFSET + idx];
        if (neutral->pid == -1)
            continue;

        mb_update_status(neutral);
        mb_update_cooldowns(neutral);
        if (neutral->stun_timer > 0)
            continue;

        mb_neutral_ai(env, neutral);
    }
}

__device__ static void mb_step_towers(Env* env) {
    for (int idx = 0; idx < NUM_TOWERS; idx++) {
        Entity* tower = &env->entities[TOWER_OFFSET + idx];
        if (tower->pid == -1)
            continue;

        mb_update_cooldowns(tower);
        if (tower->basic_attack_timer > 0)
            continue;

        if (env->tick % 3 == 0)
            mb_scan_aoe(env, tower, TOWER_VISION, true, false, false, true, true);

        Entity* target = mb_nearest_scanned_target(env, tower);
        if (target != NULL)
            mb_basic_attack(env, tower, target);
    }
}

__device__ static void mb_step_players(Env* env, const float* actions,
        float* rewards, int agent_base) {
    for (int pid = 0; pid < NUM_PLAYERS; pid++) {
        Reward* reward = &env->reward_components[pid];
        reward->death = 0;
        reward->xp = 0;
        reward->distance = 0;
        reward->tower = 0;
    }

    for (int pid = 0; pid < NUM_PLAYERS; pid++) {
        Entity* player = &env->entities[pid];
        PlayerLog* log = &env->player_logs[pid];
        Reward* reward = &env->reward_components[pid];

        if (player->mana < player->max_mana)
            player->mana += 2;
        if (player->mana > player->max_mana)
            player->mana = player->max_mana;
        if (player->health < player->max_health)
            player->health += 2;
        if (player->health > player->max_health)
            player->health = player->max_health;

        mb_update_status(player);
        mb_update_cooldowns(player);

        if (player->stun_timer > 0)
            continue;

        if (d_mcfg.script_opponents && pid >= 5) {
            mb_creep_ai(env, player);
        } else {
            const float* atn = actions + (long)(agent_base + pid) * NUM_ATNS;
            float vel_y = (atn[0] - 3.0f) / 3.0f;
            float vel_x = (atn[1] - 3.0f) / 3.0f;
            float mag = sqrtf(vel_y*vel_y + vel_x*vel_x);
            if (mag > 1) {
                vel_y /= mag;
                vel_x /= mag;
            }

            int attack_target = (int)atn[2];
            bool use_q = (int)atn[3];
            bool use_w = (int)atn[4];
            bool use_e = (int)atn[5];

            if (attack_target == 1 || attack_target == 0) {
                // Scan everything
                mb_scan_aoe(env, player, d_mcfg.vision_range, true, false, false, false, false);
            } else if (attack_target == 2) {
                // Scan only heros and towers
                mb_scan_aoe(env, player, d_mcfg.vision_range, true, false, true, true, false);
            }

            Entity* target = NULL;
            if (mb_scan_row(env, pid)[0] != -1)
                target = mb_nearest_scanned_target(env, player);

            if (use_q && player->q_timer <= 0 && mb_skill(env, player, target, 0) == 0) {
                log->usage_q += 1;
            } else if (use_w && player->w_timer <= 0 && mb_skill(env, player, target, 1) == 0) {
                log->usage_w += 1;
            } else if (use_e && player->e_timer <= 0 && mb_skill(env, player, target, 2) == 0) {
                log->usage_e += 1;
            } else if (target != NULL && mb_basic_attack(env, player, target) == 0) {
                log->usage_auto += 1;
            }

            float dest_y = player->y + player->move_modifier*d_mcfg.agent_speed*vel_y;
            float dest_x = player->x + player->move_modifier*d_mcfg.agent_speed*vel_x;
            mb_move_to(env, player, dest_y, dest_x);
        }

        float sum_reward = reward->death + reward->xp + reward->distance + reward->tower;

        log->reward_death = reward->death;
        log->reward_xp = reward->xp;
        log->reward_distance = reward->distance;
        log->reward_tower = reward->tower;
        log->episode_return += sum_reward;

        if (!d_mcfg.script_opponents || pid < 5)
            rewards[agent_base + pid] = sum_reward;
    }
}

__device__ static void mb_add_log(Env* env, float radiant_victory, float dire_victory) {
    Log* log = &env->log;
    int num_agents = d_mcfg.script_opponents ? 5 : NUM_PLAYERS;
    log->n += 1;
    log->score += radiant_victory;
    log->perf += radiant_victory;
    log->episode_length = env->tick;
    log->radiant_victory = radiant_victory;
    log->dire_victory = dire_victory;
    for (int i = 0; i < 5; i++)
        log->radiant_level += env->entities[i].level / 5.0f;
    for (int i = 5; i < 10; i++)
        log->dire_level += env->entities[i].level / 5.0f;
    for (int i = 0; i < NUM_TOWERS; i++) {
        Entity* tower = &env->entities[TOWER_OFFSET + i];
        if (TOWER_TEAM[i] == 0)
            log->radiant_towers_alive += tower->health > 0;
        else
            log->dire_towers_alive += tower->health > 0;
    }

    for (int i = 0; i < num_agents; i++) {
        PlayerLog* pl = &env->player_logs[i];
        log->episode_return += pl->episode_return;
        log->reward_death += pl->reward_death;
        log->reward_xp += pl->reward_xp;
        log->reward_distance += pl->reward_distance;
        log->reward_tower += pl->reward_tower;
    }

    PlayerLog* rs = &env->player_logs[0];
    log->radiant_support_episode_return = rs->episode_return;
    log->radiant_support_reward_death = rs->reward_death;
    log->radiant_support_reward_xp = rs->reward_xp;
    log->radiant_support_reward_distance = rs->reward_distance;
    log->radiant_support_reward_tower = rs->reward_tower;
    log->radiant_support_level = rs->level;
    log->radiant_support_kills = rs->kills;
    log->radiant_support_deaths = rs->deaths;
    log->radiant_support_damage_dealt = rs->damage_dealt;
    log->radiant_support_damage_received = rs->damage_received;
    log->radiant_support_healing_dealt = rs->healing_dealt;
    log->radiant_support_healing_received = rs->healing_received;
    log->radiant_support_creeps_killed = rs->creeps_killed;
    log->radiant_support_neutrals_killed = rs->neutrals_killed;
    log->radiant_support_towers_killed = rs->towers_killed;
    log->radiant_support_usage_auto = rs->usage_auto;
    log->radiant_support_usage_q = rs->usage_q;
    log->radiant_support_usage_w = rs->usage_w;
    log->radiant_support_usage_e = rs->usage_e;
}

// float -> unsigned char as the CPU obs buffer stores it, then -> bf16.
__device__ static inline void mb_put(obs_t* obs, int i, float v) {
    obs[i] = __float2bfloat16((float)(unsigned char)(int)v);
}

__device__ static void mb_compute_observations(Env* env, obs_t* observations, int agent_base) {
    int vis = d_mcfg.vision_range;
    for (int pid = 0; pid < d_mcfg.num_agents; pid++) {
        obs_t* obs_map = observations + (long)(agent_base + pid) * OBS_SIZE;
        for (int i = 0; i < OBS_SIZE; i++)
            obs_map[i] = __float2bfloat16(0.0f);
        obs_t* obs_extra = obs_map + 11*11*4;

        Entity* player = &env->entities[pid];
        Reward* reward = &env->reward_components[pid];

        int y = player->y;
        int x = player->x;

        mb_put(obs_extra, 0, x);
        mb_put(obs_extra, 1, y);
        mb_put(obs_extra, 2, player->level);
        mb_put(obs_extra, 3, 10*player->health/player->max_health);
        mb_put(obs_extra, 4, 10*player->mana/player->max_mana);
        mb_put(obs_extra, 5, player->damage / 50.0f);
        mb_put(obs_extra, 6, player->move_speed);
        mb_put(obs_extra, 7, player->move_modifier);
        mb_put(obs_extra, 8, player->stun_timer);
        mb_put(obs_extra, 9, player->move_timer);
        mb_put(obs_extra, 10, player->q_timer);
        mb_put(obs_extra, 11, player->w_timer);
        mb_put(obs_extra, 12, player->e_timer);
        mb_put(obs_extra, 13, player->basic_attack_timer);
        mb_put(obs_extra, 14, player->basic_attack_cd);
        mb_put(obs_extra, 15, player->is_hit);
        mb_put(obs_extra, 16, player->team);
        mb_put(obs_extra, 17 + player->hero_type, 1);

        mb_put(obs_extra, 22, (reward->death == 0) ? 0 : 1);
        mb_put(obs_extra, 23, (reward->xp == 0) ? 0 : 1);
        mb_put(obs_extra, 24, (reward->distance == 0) ? 0 : 1);
        mb_put(obs_extra, 25, (reward->tower == 0) ? 0 : 1);

        // Same (overlapping) tile layout as moba.h: write order matters.
        for (int dy = -vis; dy <= vis; dy++) {
            for (int dx = -vis; dx <= vis; dx++) {
                int map_idx = (dy + vis)*11 + (dx + vis);
                int adr = mb_offset(y + dy, x + dx);
                if (!mb_in_map(adr))
                    continue;

                mb_put(obs_map, map_idx, env->grid[adr]);
                int target_pid = env->pids[adr];
                if (target_pid == -1)
                    continue;

                Entity* target = &env->entities[target_pid];
                mb_put(obs_map, map_idx+1, 10*target->health/target->max_health);
                if (target->max_mana > 0) // Towers do not have mana
                    mb_put(obs_map, map_idx+2, 10*target->mana/target->max_mana);
                mb_put(obs_map, map_idx+3, target->level);
            }
        }
    }
}

// ---- init / reset / step ----

__device__ static void mb_init_env(Env* env) {
    env->creep_idx = 0;
    env->total_towers_taken = 0;
    env->total_levels_gained = 0;

    for (int i = 0; i < MAP_CELLS; i++) {
        env->grid[i] = d_mcfg.orig_grid[i];
        env->pids[i] = -1;
    }
    for (int i = 0; i < SCAN_ROWS; i++) {
        env->scanned[i*SCAN_COLS] = -1;
        env->scanned[i*SCAN_COLS + 1] = -1;
    }

    // hero_type order matches moba.h: support, assassin, burst, tank, carry
    const float base_health[5] = {500, 400, 400, 700, 300};
    const float base_mana[5] = {250, 300, 300, 200, 250};
    const int hp_gain[5] = {100, 100, 75, 150, 50};
    const int mana_gain[5] = {50, 65, 90, 50, 50};
    const int dmg_gain[5] = {10, 10, 10, 15, 25};
    const int lane[5] = {2, 1, 1, 0, 2};
    for (int team = 0; team < 2; team++) {
        int spawn_y = (team == 0) ? 128 - 15 : 15;
        int spawn_x = (team == 0) ? 12 : 128 - 12;
        for (int h = 0; h < 5; h++) {
            int pid = 5*team + h;
            Entity* player = &env->entities[pid];
            player->pid = pid;
            player->entity_type = ENTITY_PLAYER;
            player->team = team;
            player->spawn_y = spawn_y;
            player->spawn_x = spawn_x;
            player->x = 0;
            player->y = 0;
            player->move_speed = d_mcfg.agent_speed;
            player->basic_attack_cd = 8;
            player->base_damage = 50;
            player->last_x = 0;
            player->last_y = 0;
            player->grid_id = RADIANT_SUPPORT + h + team*5;
            player->hero_type = h;
            player->base_health = base_health[h];
            player->base_mana = base_mana[h];
            player->hp_gain_per_level = hp_gain[h];
            player->mana_gain_per_level = mana_gain[h];
            player->damage_gain_per_level = dmg_gain[h];
            player->lane = lane[h] + 3*team;
        }
    }

    for (int idx = 0; idx < NUM_TOWERS; idx++) {
        Entity* tower = &env->entities[TOWER_OFFSET + idx];
        tower->pid = TOWER_OFFSET + idx;
        tower->entity_type = ENTITY_TOWER;
        tower->grid_id = TOWER;
        tower->basic_attack_cd = 5;
        tower->team = TOWER_TEAM[idx];
        tower->spawn_y = TOWER_Y[idx];
        tower->spawn_x = TOWER_X[idx];
        tower->x = 0;
        tower->y = 0;
        tower->max_health = TOWER_HEALTH[idx];
        tower->damage = TOWER_DAMAGE[idx];
        tower->tier = TOWER_TIER[idx];
        tower->xp_on_kill = 0;
    }

    for (int camp = 0; camp < NUM_NEUTRALS/4; camp++) {
        for (int i = 0; i < 4; i++) {
            Entity* neutral = &env->entities[NEUTRAL_OFFSET + 4*camp + i];
            neutral->entity_type = ENTITY_NEUTRAL;
            neutral->grid_id = NEUTRAL;
            neutral->max_health = 500;
            neutral->team = 2;
            neutral->spawn_y = NEUTRAL_CAMP_Y[camp];
            neutral->spawn_x = NEUTRAL_CAMP_X[camp];
            neutral->x = 0;
            neutral->y = 0;
            neutral->xp_on_kill = 35;
            neutral->basic_attack_cd = 5;
            neutral->damage = 22;
        }
    }

    for (int i = 0; i < NUM_CREEPS; i++) {
        Entity* creep = &env->entities[CREEP_OFFSET + i];
        creep->pid = -1;
        creep->x = 0;
        creep->y = 0;
    }
}

__device__ static void mb_reset_env(Env* env) {
    env->tick = 0;

    for (int i = 0; i < NUM_ENTITIES; i++) {
        Entity* entity = &env->entities[i];
        entity->target_pid = -1;
        mb_scan_row(env, i)[0] = -1;
        mb_kill_entity(env, entity);
    }

    for (int i = 0; i < MAP_CELLS; i++) {
        env->grid[i] = d_mcfg.orig_grid[i];
        env->pids[i] = -1;
    }

    // Respawn towers
    for (int idx = 0; idx < NUM_TOWERS; idx++) {
        int pid = TOWER_OFFSET + idx;
        Entity* tower = &env->entities[pid];
        tower->target_pid = -1;
        tower->pid = pid;
        tower->health = tower->max_health;
        tower->basic_attack_timer = 0;
        tower->x = 0;
        tower->y = 0;
        int adr = mb_offset(tower->spawn_y, tower->spawn_x);
        env->grid[adr] = EMPTY;
        mb_move_to(env, tower, tower->spawn_y, tower->spawn_x);
        tower->last_x = tower->x;
        tower->last_y = tower->y;
    }

    // Respawn agents
    for (int i = 0; i < NUM_PLAYERS; i++) {
        env->player_logs[i] = PlayerLog{};
        Entity* player = &env->entities[i];
        player->pid = i;
        player->target_pid = -1;
        player->xp = 0;
        player->level = 1;
        mb_spawn_player(env, player);
    }
}

__device__ static void mb_step_env(Env* env, const float* actions,
        float* rewards, int agent_base) {
    for (int pid = 0; pid < NUM_ENTITIES; pid++) {
        Entity* entity = &env->entities[pid];
        entity->target_pid = -1;
        entity->attack_aoe = 0;
        entity->last_x = entity->x;
        entity->last_y = entity->y;
        entity->is_hit = 0;
    }

    mb_step_neutrals(env);
    mb_step_creeps(env);
    mb_step_towers(env);
    mb_step_players(env, actions, rewards, agent_base);
    env->tick += 1;

    float radiant_victory = 0;
    float dire_victory = 0;
    Entity* dire_ancient = &env->entities[TOWER_OFFSET + 22];
    if (dire_ancient->health <= 0 || dire_ancient->pid == -1)
        radiant_victory = 1;
    Entity* radiant_ancient = &env->entities[TOWER_OFFSET + 23];
    if (radiant_ancient->health <= 0 || radiant_ancient->pid == -1)
        dire_victory = 1;

    if (radiant_victory || dire_victory) {
        mb_add_log(env, radiant_victory, dire_victory);
        mb_reset_env(env);
    }
}

__global__ void mb_init_kernel(Env* envs, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    mb_init_env(&envs[i]);
}

__global__ void mb_reset_kernel(Env* envs, obs_t* observations,
        float* rewards, float* terminals, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    int agent_base = i * d_mcfg.num_agents;
    mb_reset_env(&envs[i]);
    mb_compute_observations(&envs[i], observations, agent_base);
    for (int a = 0; a < d_mcfg.num_agents; a++) {
        rewards[agent_base + a] = 0.0f;
        terminals[agent_base + a] = 0.0f;
    }
}

// moba.h never sets terminals (episodes reset internally), so neither do we.
__global__ void mb_step_kernel(Env* __restrict__ envs, const float* __restrict__ actions,
        obs_t* __restrict__ observations, float* __restrict__ rewards, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    int agent_base = i * d_mcfg.num_agents;
    mb_step_env(&envs[i], actions, rewards, agent_base);
    mb_compute_observations(&envs[i], observations, agent_base);
}

// ---- host ----

// Host copy of moba.h's bfs (only walls block), used to build the shared table.
static void mb_host_bfs(const unsigned char* grid, unsigned char* paths, int* buffer,
        int dest_r, int dest_c) {
    const int N = MAP_N;
    if (grid[dest_r*N + dest_c] == WALL)
        return;
    const int dr[8] = {-1, 1, 0, 0, -1, 1, 1, -1};
    const int dc[8] = {0, 0, -1, 1, 1, 1, -1, -1};
    int start = 0;
    int end = 1;
    buffer[0] = 0;
    buffer[1] = dest_r;
    buffer[2] = dest_c;
    while (start < end) {
        int atn = buffer[3*start];
        int r = buffer[3*start + 1];
        int c = buffer[3*start + 2];
        start++;
        if (r < 0 || r >= N || c < 0 || c >= N)
            continue;
        if (paths[r*N + c] != 255)
            continue;
        if (grid[r*N + c] == WALL) {
            paths[r*N + c] = 8;
            continue;
        }
        paths[r*N + c] = atn;
        for (int k = 0; k < 8; k++) {
            buffer[3*end] = k;
            buffer[3*end + 1] = r + dr[k];
            buffer[3*end + 2] = c + dc[k];
            end++;
        }
    }
    paths[dest_r*N + dest_c] = 8;
}

static unsigned char* mb_host_precompute_paths(const unsigned char* grid) {
    const size_t total = (size_t)MAP_CELLS * MAP_CELLS;
    unsigned char* paths = (unsigned char*)malloc(total);
    memset(paths, 255, total);
    #pragma omp parallel
    {
        int* buffer = (int*)calloc(3*8*MAP_CELLS, sizeof(int));
        #pragma omp for schedule(dynamic, 16)
        for (int dst = 0; dst < MAP_CELLS; dst++) {
            mb_host_bfs(grid, &paths[(size_t)dst * MAP_CELLS], buffer, dst / MAP_N, dst % MAP_N);
        }
        free(buffer);
    }
    return paths;
}

static struct {
    Env* envs;
    int n;
    int num_agents;
    obs_t* observations;
    float* actions;
    float* rewards;
    float* terminals;
    cudaStream_t stream;
    unsigned char* d_paths;
    unsigned char* d_grid;
    int window_open;
} g_gpu;

static int mb_grid(int n) {
    return (n + MOBA_BLOCK - 1) / MOBA_BLOCK;
}

Env* puf_vec_create(int total_agents, Dict* env_kwargs,
        obs_t* observations, float* actions, float* rewards, float* terminals) {
    MobaConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.vision_range = dict_get(env_kwargs, "vision_range");
    cfg.agent_speed = dict_get(env_kwargs, "agent_speed");
    cfg.script_opponents = dict_get(env_kwargs, "script_opponents");
    cfg.reward_death = dict_get(env_kwargs, "reward_death");
    cfg.reward_xp = dict_get(env_kwargs, "reward_xp");
    cfg.reward_distance = dict_get(env_kwargs, "reward_distance");
    cfg.reward_tower = dict_get(env_kwargs, "reward_tower");
    cfg.num_agents = cfg.script_opponents ? 5 : 10;
    if (total_agents % cfg.num_agents != 0) {
        fprintf(stderr, "moba: total_agents must be a multiple of %d\n", cfg.num_agents);
        exit(1);
    }
    if (cfg.vision_range > 5) {
        fprintf(stderr, "moba: vision_range > 5 overflows the 11x11 obs window\n");
        exit(1);
    }

    unsigned char* h_paths = mb_host_precompute_paths(game_map_npy);
    cudaMalloc((void**)&g_gpu.d_paths, (size_t)MAP_CELLS * MAP_CELLS);
    cudaMemcpy(g_gpu.d_paths, h_paths, (size_t)MAP_CELLS * MAP_CELLS, cudaMemcpyHostToDevice);
    free(h_paths);
    cudaMalloc((void**)&g_gpu.d_grid, MAP_CELLS);
    cudaMemcpy(g_gpu.d_grid, game_map_npy, MAP_CELLS, cudaMemcpyHostToDevice);
    cfg.paths = g_gpu.d_paths;
    cfg.orig_grid = g_gpu.d_grid;
    cudaMemcpyToSymbol(d_mcfg, &cfg, sizeof(cfg));

    int n = total_agents / cfg.num_agents;
    Env* host_envs = (Env*)calloc(n, sizeof(Env));
    for (int i = 0; i < n; i++) {
        host_envs[i].rng = (unsigned int)(i + 1) * 2654435761u;
        host_envs[i].num_agents = cfg.num_agents;
    }
    Env* envs = NULL;
    cudaMalloc((void**)&envs, (size_t)n * sizeof(Env));
    cudaMemcpy(envs, host_envs, (size_t)n * sizeof(Env), cudaMemcpyHostToDevice);
    free(host_envs);
    mb_init_kernel<<<mb_grid(n), MOBA_BLOCK>>>(envs, n);

    g_gpu.envs = envs;
    g_gpu.n = n;
    g_gpu.num_agents = cfg.num_agents;
    g_gpu.observations = observations;
    g_gpu.actions = actions;
    g_gpu.rewards = rewards;
    g_gpu.terminals = terminals;
    g_gpu.stream = 0;
    g_gpu.window_open = 0;
    return envs;
}

void puf_bind_stream(cudaStream_t stream) {
    g_gpu.stream = stream;
}

void puf_init(Env* env, Dict* kwargs) {
    env->num_agents = dict_get(kwargs, "script_opponents") ? 5 : 10;
}

void puf_reset(Env* env) {
    mb_reset_kernel<<<mb_grid(g_gpu.n), MOBA_BLOCK>>>(
        g_gpu.envs, g_gpu.observations, g_gpu.rewards, g_gpu.terminals, g_gpu.n);
}

void puf_step(Env* env) {
    mb_step_kernel<<<mb_grid(g_gpu.n), MOBA_BLOCK, 0, g_gpu.stream>>>(
        g_gpu.envs, g_gpu.actions, g_gpu.observations, g_gpu.rewards, g_gpu.n);
}

void puf_close(Env* env) {
    if (g_gpu.window_open) {
        CloseWindow();
        g_gpu.window_open = 0;
    }
    cudaFree(g_gpu.envs);
    cudaFree(g_gpu.d_paths);
    cudaFree(g_gpu.d_grid);
    g_gpu.envs = NULL;
}

// Minimal map view of env 0 (moba.h has the full renderer).
void puf_render(Env* env) {
    if (!g_gpu.envs || g_gpu.n < 1)
        return;
    if (g_gpu.stream)
        cudaStreamSynchronize(g_gpu.stream);
    static unsigned char grid[MAP_CELLS];
    cudaMemcpy(grid, (char*)g_gpu.envs + offsetof(Env, grid), MAP_CELLS, cudaMemcpyDeviceToHost);

    const int cell = 6;
    if (!g_gpu.window_open) {
        InitWindow(MAP_N*cell, MAP_N*cell, "PufferLib Moba (GPU)");
        SetTargetFPS(30);
        g_gpu.window_open = 1;
    }
    if (IsKeyDown(KEY_ESCAPE))
        exit(0);

    const Color colors[] = {
        {6, 24, 24, 255}, {0, 178, 178, 255}, {255, 165, 0, 255},
        {0, 0, 128, 255}, {128, 0, 0, 255}, {128, 128, 128, 255},
    };
    BeginDrawing();
    ClearBackground(colors[0]);
    for (int y = 0; y < MAP_N; y++) {
        for (int x = 0; x < MAP_N; x++) {
            int tile = grid[y*MAP_N + x];
            if (tile == EMPTY)
                continue;
            Color c = tile <= NEUTRAL ? colors[tile]
                : (tile <= RADIANT_CARRY ? (Color){0, 0, 255, 255} : (Color){255, 0, 0, 255});
            DrawRectangle(x*cell, y*cell, cell, cell, c);
        }
    }
    EndDrawing();
    puf_web_vsync();
}

#endif
