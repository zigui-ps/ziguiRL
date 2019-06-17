#include<stdio.h>
#include<stdlib.h>
#include<math.h>

#define BALL 10

struct State{
	double s[BALL], v[BALL], a[BALL];
};

struct Action{
	Action(double a): a(a){}
	double a;
};

struct Result{
	Result(State s, double r, bool done, bool time_limit):
		state(s), reward(r), done(done), time_limit(time_limit){}
	State state;
	double reward;
	bool done, time_limit;
};

int steps = 0;
State cur;

void render()
{
	printf("step #%d\n", steps);
	for(int i = 0; i < BALL; i++){
		printf("ball %2d : %.3lf ; %.3lf ; %.3lf\n", i, cur.s[i], cur.v[i], cur.a[i]);
	}
	printf("\n");
}

State reset()
{
	steps = 0;
	for(int i = 0; i < BALL; i++){
		cur.s[i] = rand() / (double)RAND_MAX / 10;
		cur.v[i] = 0;
		cur.a[i] = 0;
	}
	return cur;
}

Result step(Action action)
{
	steps += 1;
	for(int i = 0; i < BALL; i++) cur.a[i] += action.a;
	for(int i = 0; i < BALL; i++) cur.v[i] += cur.a[i] / 10.0;
	for(int i = 0; i < BALL; i++) cur.s[i] += cur.v[i] / 10.0;

	double reward = 0;
	for(int i = 0; i < BALL; i++){
		reward += (10 - fabs(cur.s[i])) * (10 - fabs(cur.s[i])) / 100.0;
	}

	bool done = false;
	for(int i = 0; i < BALL; i++) if(fabs(cur.s[i]) > 10) done = true;
	if(done) reward = 0;

	bool time_limit = steps >= 1000;
	if(time_limit) done = true;

	return Result(cur, reward, done, time_limit);
}
