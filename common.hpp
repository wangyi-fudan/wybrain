#ifndef	wybrain_common
#define	wybrain_common
#include	<stdlib.h>
#include	<string.h>
#include	"wyhash.h"
#include	<float.h>
#include	<stdio.h>
#include	<math.h>
#include	<time.h>
uint64_t	wybrain_seed=wyhash64(time(NULL),rand());
float	learning_rate=0.1;

template<unsigned	input,	unsigned	output,	class	type=float>
struct	matrix{
	type	weight[input*output];
	matrix(){	for(unsigned    i=0;    i<input*output; i++)	weight[i]=wy2gau(wyrand(&wybrain_seed));	}
	type*	operator()(unsigned	i){	return	weight+i*output;	}
	void	save(FILE	*f){	fwrite(weight,sizeof(type),input*output,f);	}
	bool	load(FILE	*f){	return	fread(weight,sizeof(type),input*output,f)==input*output;	}
};
#endif
