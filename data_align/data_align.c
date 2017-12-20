#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef struct {
	int k;
	int c;
	float m;	
} a;

typedef struct {
	float k;
	float c;
	float m;	
} b;

int main() {
	float *k;
	k = malloc(sizeof(a) + sizeof(b));
	a k1;
	k1.k = 1; k1.c = 2; k1.m = 3;
	b k2;
	k2.k = 4; k2.c = 5; k2.m = 6;
	memcpy(k, &k1, sizeof(a));
	memcpy(k + sizeof(a), &k2, sizeof(b));
	
	a *p_k1; b *p_k2;
	p_k1 = (a*)k;
	p_k2 = (b*)(k + sizeof(a));

	printf("p_k1->k = %d\n", p_k1->k);	
	printf("p_k2->k = %f\n", p_k2->k);
	return 0;
}
