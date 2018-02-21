#ifndef _PATHFINDER_VAR_H_
#define _PATHFINDER_VAR_H_

#define BLOCK_SIZE_PATH_FINDER 256
#define STR_SIZE_PATH_FINDER 256
#define HALO_PATH_FINDER 1 // halo width along one direction when advancing to the next iteration

#define IN_RANGE_PATH_FINDER(x, min, max)   ((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE_PATH_FINDER(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )
#define MIN_PATH_FINDER(a, b) ((a)<=(b) ? (a) : (b))
#define M_SEED_PATH_FINDER 9

#endif