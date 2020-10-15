#include	"common.hpp"
template<uint64_t	input,	uint64_t	output,	uint64_t	batch=1,	class	actfun=af_default>
struct	fc_hidden{
	matrix<input+1,output>	w;
	matrix<batch,input>	g;
	matrix<batch,output>	o;
	void	forward(const	float	*inp,	uint64_t	b=0){
		float	*out=o(b);
		memset(out,0,output*sizeof(float));	
		for(uint64_t	i=0;	i<=input;	i++){
			float	s=i<input?inp[i]:1,	*p=w(i);
			for(uint64_t	j=0;	j<output;	j++)	out[j]+=s*p[j];
		}
		const	float	sca=1/sqrt(input+1);	actfun	af;	
		for(uint64_t	i=0;	i<output;	i++)	out[i]=af.act(sca*out[i]);
	}
	void	backward(const	float	*inp,	const	float	*bac,	uint64_t	b=0){
		float	*gra=g(b),	*out=o(b);
		const	float	sca=1/sqrtf(input+1);	actfun	af;
		for(uint64_t	i=0;	i<output;	i++)	out[i]=af.gra(out[i])*bac[i]*sca;
		for(uint64_t	i=0;	i<=input;	i++){
			float	s=(i<input?inp[i]:1),	*p=w(i),	z=0;
			for(uint64_t	j=0;	j<output;	j++){	z+=p[j]*out[j];	p[j]-=s*out[j];	}
			if(i<input)	gra[i]=z;
		}
	}
};

template<uint64_t	input,	uint64_t	output,	uint64_t	bits,	uint64_t	batch=1,	class	actfun=af_default>
struct	fc_sparse{
	matrix_large<(input+1)<<bits,output>	w;
	matrix<bits,input>	h;
	matrix<batch,input>	g;
	matrix<batch,output>	o;
	uint64_t	k[batch];
	void	forward(const	float	*inp,	uint64_t	b=0){
		k[b]=0;
		for(uint64_t	i=0;	i<bits;	i++){
			float	*p=h(i),	s=0;
			for(uint64_t	j=0;	j<input;	j++)	s+=p[j]*inp[j];
			if(s>0)	k[b]|=1ull<<i;
		}
		float	*out=o(b);
		memset(out,0,output*sizeof(float));	
		for(uint64_t	i=0;	i<=input;	i++){
			float	s=i<input?inp[i]:1,	*p=w(k[b]*(input+1)+i);
			for(uint64_t	j=0;	j<output;	j++)	out[j]+=s*p[j];
		}
		const	float	sca=1/sqrt(input+1);	actfun	af;	
		for(uint64_t	i=0;	i<output;	i++)	out[i]=af.act(sca*out[i]);
	}
	void	backward(const	float	*inp,	const	float	*bac,	uint64_t	b=0){
		float	*gra=g(b),	*out=o(b);
		const	float	sca=1/sqrtf(input+1);	actfun	af;
		for(uint64_t	i=0;	i<output;	i++)	out[i]=af.gra(out[i])*bac[i]*sca;
		for(uint64_t	i=0;	i<=input;	i++){
			float	s=(i<input?inp[i]:1),	*p=w(k[b]*(input+1)+i),	z=0;
			for(uint64_t	j=0;	j<output;	j++){	z+=p[j]*out[j];	p[j]-=s*out[j];	}
			if(i<input)	gra[i]=z;
		}
	}
};

