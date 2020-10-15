#include	"common.hpp"
template<uint64_t	input,	uint64_t	size,	uint64_t	batch=1>
struct	me_hidden{
	matrix<size,input>	w;
	matrix<batch,input>	g;
	matrix<batch,input>	o;
	matrix<batch,size>	v;
	void	forward(const	float	*inp,	uint64_t	b=0){
		float	*val=v(b),	*out=o(b),	ma=-FLT_MAX,	sum=0;
		const	float	sca=1/sqrt(input);
		memset(out,	0,	input*sizeof(float));
		for(uint64_t	i=0;	i<size;	i++){
			float	s=0,	*p=w(i);
			for(uint64_t	j=0;	j<input;	j++)	s+=inp[j]*p[j];
			val[i]=s*sca;
			if(val[i]>ma)	ma=val[i];
		}
		for(uint64_t	i=0;	i<size;	i++)	sum+=(val[i]=expf(val[i]-ma));
		for(uint64_t	i=0;	i<size;	i++){
			val[i]/=sum;	float	s=val[i],	*p=w(i);
			for(uint64_t	j=0;	j<input;	j++)	out[j]+=s*p[j];
		}
	}
	void	backward(const	float	*inp,	const	float	*bac,	uint64_t	b=0){
		float	*gra=g(b),	*out=o(b),	*val=v(b);
		const	float	sca=1/sqrtf(input);
		memset(gra,	0,	input*sizeof(float));
		for(uint64_t	i=0;	i<size;	i++){
			float	*p=w(i),	s=val[i];
			for(uint64_t	j=0;	j<input;	j++){
				gra[j]+=bac[j]*sca*s*p[j]*(p[j]-out[j]);
				p[j]-=bac[j]*s;
			}
		}
	}
};

template<uint64_t	input,	uint64_t	size,	uint64_t	bits,	uint64_t	batch=1>
struct	me_sparse{
	matrix_large<size<<bits,input>	w;
	matrix<bits,input>	h;
	matrix<batch,input>	g;
	matrix<batch,input>	o;
	matrix<batch,size>	v;
	uint64_t	k[batch];
	void	forward(const	float	*inp,	uint64_t	b=0){
		float	*val=v(b),	*out=o(b),	ma=-FLT_MAX,	sum=0;
		const	float	sca=1/sqrt(input);
		k[b]=0;
		for(uint64_t	i=0;	i<bits;	i++){
			float	*p=h(i),	s=0;
			for(uint64_t	j=0;	j<input;	j++)	s+=p[j]*inp[j];
			if(s>0)	k[b]|=1ull<<i;
		}
		memset(out,	0,	input*sizeof(float));
		for(uint64_t	i=0;	i<size;	i++){
			float	s=0,	*p=w(k[b]*size+i);
			for(uint64_t	j=0;	j<input;	j++)	s+=inp[j]*p[j];
			val[i]=s*sca;
			if(val[i]>ma)	ma=val[i];
		}
		for(uint64_t	i=0;	i<size;	i++)	sum+=(val[i]=expf(val[i]-ma));
		for(uint64_t	i=0;	i<size;	i++){
			val[i]/=sum;	float	s=val[i],	*p=w(k[b]*size+i);
			for(uint64_t	j=0;	j<input;	j++)	out[j]+=s*p[j];
		}
	}
	void	backward(const	float	*inp,	const	float	*bac,	uint64_t	b=0){
		float	*gra=g(b),	*out=o(b),	*val=v(b);
		const	float	sca=1/sqrtf(input);
		memset(gra,	0,	input*sizeof(float));
		for(uint64_t	i=0;	i<size;	i++){
			float	*p=w(k[b]*size+i),	s=val[i];
			for(uint64_t	j=0;	j<input;	j++){
				gra[j]+=bac[j]*sca*s*p[j]*(p[j]-out[j]);
				p[j]-=bac[j]*s;
			}
		}
	}
};

