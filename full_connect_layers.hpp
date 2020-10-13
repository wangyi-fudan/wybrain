#include	"common.hpp"
template<unsigned	input,	unsigned	output,	unsigned	batch=1,	class	actfun=af_default>
struct	fc_hidden{
	matrix<input+1,output>	w;
	matrix<batch,input>	g;
	matrix<batch,output>	o;
	void	forward(const	float	*inp,	unsigned	b=0){
		float	*out=o(b);
		memset(out,0,output*sizeof(float));	
		for(unsigned	i=0;	i<=input;	i++){
			float	s=i<input?inp[i]:1,	*p=w(i);
			for(unsigned	j=0;	j<output;	j++)	out[j]+=s*p[j];
		}
		const	float	sca=1/sqrt(input+1);	actfun	af;	
		for(unsigned	i=0;	i<output;	i++){
			out[i]=af.act(sca*out[i]);
		}
	}
	void	backward(const	float	*inp,	const	float	*gradient,	float	eta,	unsigned	b=0){
		float	*gra=g(b),	*out=o(b);
		const	float	sca=1/sqrtf(input+1);	actfun	af;
		for(unsigned	i=0;	i<output;	i++)	out[i]=af.gra(out[i])*gradient[i]*sca;
		for(unsigned	i=0;	i<=input;	i++){
			float	s=(i<input?inp[i]:1)*eta,	*p=w(i),	z=0;
			for(unsigned	j=0;	j<output;	j++){	z+=p[j]*out[j];	p[j]-=s*out[j];	}
			if(i<input)	gra[i]=z;
		}
	}
};

