#include	"common.hpp"
template<uint64_t	input,	uint64_t	output,	uint64_t	batch=1,	class	actfun=af_default>
struct	em_dense{
	matrix<input+1,output>	w;
	matrix<batch,output>	o;
	uint64_t	seed[batch];
	void	forward(const	float	*inp,	float	dropout,	uint64_t	b=0){
		float	*out=o(b);	seed[b]=wybrain_seed;
		memset(out,0,output*sizeof(float));	
		for(uint64_t	i=0;	i<=input;	i++){
			float	s=(i<input?inp[i]:1)*(dropout>0?wy2u01(wyrand(&wybrain_seed))>dropout:1+dropout),	*p=w(i);
			for(uint64_t	j=0;	j<output;	j++)	out[j]+=s*p[j];
		}
		const	float	sca=1/sqrt(input+1);	actfun	af;
		for(uint64_t	i=0;	i<output;	i++)	out[i]=af.act(sca*out[i]);
	}
	void	backward(const	float	*inp,	const	float	*bac,	float	dropout,	uint64_t	b=0){
		float	*out=o(b);
		const	float	sca=1/sqrtf(input+1);	actfun	af;
		for(uint64_t	i=0;	i<output;	i++)	out[i]=af.gra(out[i])*bac[i]*sca;
		for(uint64_t	i=0;	i<=input;	i++){
			float	s=(i<input?inp[i]:1)*(dropout>0?wy2u01(wyrand(&seed[b]))>dropout:1+dropout),	*p=w(i);
			for(uint64_t	j=0;	j<output;	j++)	p[j]-=s*out[j];
		}
	}
};

template<uint64_t	input,	uint64_t	output,	uint64_t	batch=1,	class	actfun=af_default>
struct	em_binary{
	matrix<input+1,output>	w;
	matrix<batch,output>	o;
	uint64_t	seed[batch];
	void	forward(const	void	*inp,	float	dropout,	uint64_t	b=0){
		float	*out=o(b);	seed[b]=wybrain_seed;
		memset(out,0,output*sizeof(float));	const	uint8_t	*dat=(const	uint8_t*)inp;
		for(uint64_t	i=0;	i<=input;	i++){
			float	s=(i<input?(((int)((dat[i>>3]>>(i&7))&1))<<1)-1:1)*(dropout>0?wy2u01(wyrand(&wybrain_seed))>dropout:1+dropout),	*p=w(i);
			for(uint64_t	j=0;	j<output;	j++)	out[j]+=s*p[j];
		}
		const	float	sca=1/sqrt(input+1);	actfun	af;
		for(uint64_t	i=0;	i<output;	i++)	out[i]=af.act(sca*out[i]);
	}
	void	backward(const	void	*inp,	const	float	*bac,	float	dropout,	uint64_t	b=0){
		float	*out=o(b);
		const	float	sca=1/sqrtf(input+1);	actfun	af;	const   uint8_t *dat=(const uint8_t*)inp;
		for(uint64_t	i=0;	i<output;	i++)	out[i]=af.gra(out[i])*bac[i]*sca;
		for(uint64_t	i=0;	i<=input;	i++){
			float	s=(i<input?(((int)((dat[i>>3]>>(i&7))&1))<<1)-1:1)*(dropout>0?wy2u01(wyrand(&seed[b]))>dropout:1+dropout),	*p=w(i);
			for(uint64_t	j=0;	j<output;	j++)	p[j]-=s*out[j];
		}
	}
};

template<uint64_t	input,	uint64_t	output,	uint64_t	batch=1,	class	actfun=af_default>
struct	em_sparse{
	matrix<input+1,output>	w;
	matrix<batch,output>	o;
	uint64_t	seed[batch];
	void	forward(const	uint64_t	*inp,	uint64_t	n,	float	dropout,	uint64_t	b=0){
		float	*out=o(b);	seed[b]=wybrain_seed;
		memset(out,0,output*sizeof(float));
		for(uint64_t	i=0;	i<=n;	i++){
			float	*p=w(i<n?inp[i]:input),	s=(dropout>0?wy2u01(wyrand(&wybrain_seed))>dropout:1+dropout);
			for(uint64_t	j=0;	j<output;	j++)	out[j]+=s*p[j];
		}
		const	float	sca=1/sqrt(n+1);	actfun	af;
		for(uint64_t	i=0;	i<output;	i++)	out[i]=af.act(sca*out[i]);
	}
	void	backward(const	uint64_t	*inp,	uint64_t	n,	const	float	*bac,	float	dropout,	uint64_t	b=0){
		float	*out=o(b);
		const	float	sca=1/sqrtf(n+1);	actfun	af;
		for(uint64_t	i=0;	i<output;	i++)	out[i]=af.gra(out[i])*bac[i]*sca;
		for(uint64_t	i=0;	i<=n;	i++){
			float	*p=w(i<n?inp[i]:input),	s=(dropout>0?wy2u01(wyrand(&seed[b]))>dropout:1+dropout);
			for(uint64_t	j=0;	j<output;	j++)	p[j]-=s*out[j];
		}
	}
};
