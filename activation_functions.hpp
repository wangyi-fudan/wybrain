#ifndef	wybrain_activation_functions
#define	wybrain_activation_functions
#include	"common.hpp"

struct	af_linear{
	float	act(float	x){	return	x;	}
	float	gra(float	x){	return	1;	}
};

struct	af_isru{
	float	act(float	x){	return	x/sqrtf(1+x*x);	}
	float	gra(float	x){	x=1-x*x;	return	x*sqrtf(x);	}
};

struct	af_softsign{
	float	act(float	x){	return	x/(1+fabsf(x));	}
	float	gra(float	x){	x=1-fabsf(x);	return	x*x;	}
};

struct	af_relu{
	float	act(float	x){	return	x*(x>0);	}
	float	gra(float	x){	return	x>0;	}
};

struct	af_leaky_relu{
	float	act(float	x){	return	x>0?x:0.01f*x;	}
	float	gra(float	x){	return	x>0?1:0.01f;	}
};

struct	af_sigmoid{
	float	act(float	x){	return	1/(1+expf(-x));	}
	float	gra(float	x){	return	x*(1-x);	}
};

struct	af_softplus{
	float	act(float	x){	return	logf(1+expf(x));	}
	float	gra(float	x){	return	1-expf(-x);	}
};


struct	af_tanh{
	float	act(float	x){	return	tanhf(x);	}
	float	gra(float	x){	return	1-x*x;	}
};

struct	af_hardtanh{
	float	act(float	x){	return	x>1?1:(x<-1?-1:x);	}
	float	gra(float	x){	return	x>=1||x<=-1?0:1;	}
};

struct	af_atan{
	float	act(float	x){	return	atanf(x);	}
	float	gra(float	x){	x=cosf(x);;	return	x*x;	}
};

typedef	af_isru	af_default;
#endif
