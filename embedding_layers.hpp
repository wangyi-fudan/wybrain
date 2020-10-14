#include	"common.hpp"
template<unsigned	input,	unsigned	output,	unsigned	batch=1,	class	actfun=af_default>
struct	embed_dense{
	matrix<input+1,output>	w;
	matrix<batch,output>	o;
	void	forward(const	float	*inp,	unsigned	b=0){
		float	*out=o(b);
		memset(out,0,output*sizeof(float));	
		for(unsigned	i=0;	i<=input;	i++){
			float	s=i<input?inp[i]:1,	*p=w(i);
			for(unsigned	j=0;	j<output;	j++)	out[j]+=s*p[j];
		}
		const	float	sca=1/sqrt(input+1);	actfun	af;
		for(unsigned	i=0;	i<output;	i++)	out[i]=af.act(sca*out[i]);
	}
	void	backward(const	float	*inp,	const	float	*bac,	unsigned	b=0){
		float	*out=o(b);
		const	float	sca=1/sqrtf(input+1);	actfun	af;
		for(unsigned	i=0;	i<output;	i++)	out[i]=af.gra(out[i])*bac[i]*sca;
		for(unsigned	i=0;	i<=input;	i++){
			float	s=(i<input?inp[i]:1),	*p=w(i);
			for(unsigned	j=0;	j<output;	j++)	p[j]-=s*out[j];
		}
	}
};

template<unsigned	input,	unsigned	output,	unsigned	batch=1,	class	actfun=af_default>
struct	embed_binary{
	matrix<input+1,output>	w;
	matrix<batch,output>	o;
	void	forward(const	void	*inp,	unsigned	b=0){
		float	*out=o(b);
		memset(out,0,output*sizeof(float));	const	uint8_t	*dat=(const	uint8_t*)inp;
		for(unsigned	i=0;	i<=input;	i++){
			float	s=i<input?(((int)((dat[i>>3]>>(i&7))&1))<<1)-1:1,	*p=w(i);
			for(unsigned	j=0;	j<output;	j++)	out[j]+=s*p[j];
		}
		const	float	sca=1/sqrt(input+1);	actfun	af;
		for(unsigned	i=0;	i<output;	i++)	out[i]=af.act(sca*out[i]);
	}
	void	backward(const	void	*inp,	const	float	*bac,	unsigned	b=0){
		float	*out=o(b);
		const	float	sca=1/sqrtf(input+1);	actfun	af;	const   uint8_t *dat=(const uint8_t*)inp;
		for(unsigned	i=0;	i<output;	i++)	out[i]=af.gra(out[i])*bac[i]*sca;
		for(unsigned	i=0;	i<=input;	i++){
			float	s=(i<input?(((int)((dat[i>>3]>>(i&7))&1))<<1)-1:1),	*p=w(i);
			for(unsigned	j=0;	j<output;	j++)	p[j]-=s*out[j];
		}
	}
};

template<unsigned	input,	unsigned	output,	unsigned	batch=1,	class	actfun=af_default>
struct	embed_sparse{
	matrix<input+1,output>	w;
	matrix<batch,output>	o;
	void	forward(const	unsigned	*inp,	unsigned	n,	unsigned	b=0){
		float	*out=o(b);
		memset(out,0,output*sizeof(float));
		for(unsigned	i=0;	i<=n;	i++){
			float	*p=w(i<n?inp[i]:input);
			for(unsigned	j=0;	j<output;	j++)	out[j]+=p[j];
		}
		const	float	sca=1/sqrt(n+1);	actfun	af;
		for(unsigned	i=0;	i<output;	i++)	out[i]=af.act(sca*out[i]);
	}
	void	backward(const	unsigned	*inp,	unsigned	n,	const	float	*bac,	unsigned	b=0){
		float	*out=o(b);
		const	float	sca=1/sqrtf(n+1);	actfun	af;
		for(unsigned	i=0;	i<output;	i++)	out[i]=af.gra(out[i])*bac[i]*sca;
		for(unsigned	i=0;	i<=n;	i++){
			float	*p=w(i<n?inp[i]:input);
			for(unsigned	j=0;	j<output;	j++)	p[j]-=out[j];
		}
	}
};
