#include	"common.hpp"
template<unsigned	input,	unsigned	size,	unsigned	batch=1>
struct	me_hidden{
	matrix<size,input>	w;
	matrix<batch,input>	g;
	matrix<batch,input>	o;
	matrix<batch,size>	v;
	void	forward(const	float	*inp,	unsigned	b=0){
		float	*val=v(b),	*out=o(b),	ma=-FLT_MAX,	sum=0;
		const	float	sca=1/sqrt(input);
		memset(out,	0,	input*sizeof(float));
		for(unsigned	i=0;	i<size;	i++){
			float	s=0,	*p=w(i);
			for(unsigned	j=0;	j<input;	j++)	s+=inp[j]*p[j];
			val[i]=s*sca;
			if(val[i]>ma)	ma=val[i];
		}
		for(unsigned	i=0;	i<size;	i++)	sum+=(val[i]=expf(val[i]-ma));
		for(unsigned	i=0;	i<size;	i++){
			val[i]/=sum;	float	s=val[i],	*p=w(i);
			for(unsigned	j=0;	j<input;	j++)	out[j]+=s*p[j];
		}
	}
	void	backward(const	float	*inp,	const	float	*bac,	unsigned	b=0){
		float	*gra=g(b),	*out=o(b),	*val=v(b);
		const	float	sca=1/sqrtf(input);
		memset(gra,	0,	input*sizeof(float));
		for(unsigned	i=0;	i<size;	i++){
			float	*p=w(i),	s=val[i];
			for(unsigned	j=0;	j<input;	j++){
				gra[j]+=bac[j]*sca*s*p[j]*(p[j]-out[j]);
				p[j]-=bac[j]*s;
			}
		}
	}
};

