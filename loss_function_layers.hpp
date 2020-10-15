#include	"common.hpp"
template<uint64_t	input,	uint64_t	output,	uint64_t	batch=1>
struct	lf_l2{
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
		const	float	sca=1/sqrtf(input+1);
		for(uint64_t	i=0;	i<output;	i++)	out[i]*=sca;
	}
	float	backward(const	float	*inp,	const	float	*target,	uint64_t	b=0){
		float	*gra=g(b),	*out=o(b);
		const	float	sca=1/sqrtf(input+1);	float	l=0;
		for(uint64_t	i=0;	i<output;	i++){	out[i]-=target[i];	l+=out[i]*out[i];	out[i]*=sca*learning_rate;	}
		for(uint64_t	i=0;	i<=input;	i++){
			float	s=(i<input?inp[i]:1),	*p=w(i),	z=0;
			for(uint64_t	j=0;	j<output;	j++){	z+=p[j]*out[j];	p[j]-=s*out[j];	}
			if(i<input)	gra[i]=z;
		}
		return	l;
	}
};

template<uint64_t	input,	uint64_t	output,	uint64_t	batch=1>
struct	lf_l1{
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
		const	float	sca=1/sqrtf(input+1);
		for(uint64_t	i=0;	i<output;	i++)	out[i]*=sca;
	}
	float	backward(const	float	*inp,	const	float	*target,	uint64_t	b=0){
		float	*gra=g(b),	*out=o(b);
		const	float	sca=1/sqrtf(input+1);	float	l=0;
		for(uint64_t	i=0;	i<output;	i++){	l+=fabsf(out[i]-target[i]);	out[i]=(out[i]>target[i]?1:-1)*sca*learning_rate;	}
		for(uint64_t	i=0;	i<=input;	i++){
			float	s=(i<input?inp[i]:1),	*p=w(i),	z=0;
			for(uint64_t	j=0;	j<output;	j++){	z+=p[j]*out[j];	p[j]-=s*out[j];	}
			if(i<input)	gra[i]=z;
		}
		return	l;
	}
};

template<uint64_t	input,	uint64_t	output,	uint64_t	batch=1>
struct	lf_logistic{
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
		const	float	sca=1/sqrtf(input+1);
		for(uint64_t	i=0;	i<output;	i++)	out[i]=1/(1+expf(-sca*out[i]));
	}
	float	backward(const	float	*inp,	const	float	*target,	uint64_t	b=0){
		float	*gra=g(b),	*out=o(b);
		const	float	sca=1/sqrtf(input+1);	float	l=0;
		for(uint64_t	i=0;	i<output;	i++){
			l-=target[i]*logf(fmaxf(out[i],FLT_MIN))-(1-target[i])*logf(fmaxf(1-out[i],FLT_MIN));
			out[i]=(out[i]-target[i])*sca*learning_rate;
		}
		for(uint64_t	i=0;	i<=input;	i++){
			float	s=(i<input?inp[i]:1),	*p=w(i),	z=0;
			for(uint64_t	j=0;	j<output;	j++){	z+=p[j]*out[j];	p[j]-=s*out[j];	}
			if(i<input)	gra[i]=z;
		}
		return	l;
	}
};

template<uint64_t	input,	uint64_t	output,	uint64_t	batch=1>
struct	lf_softmax{
	matrix<input+1,output>	w;
	matrix<batch,input>	g;
	matrix<batch,output>	o;
	void	forward(const	float	*inp,	float	alpha=1,	uint64_t	b=0){
		float	*out=o(b);
		memset(out,0,output*sizeof(float));
		for(uint64_t	i=0;	i<=input;	i++){
			float	s=i<input?inp[i]:1,	*p=w(i);
			for(uint64_t	j=0;	j<output;	j++)	out[j]+=s*p[j];
		}
		const	float	sca=1/sqrtf(input+1);	float	ma=-FLT_MAX,	sum=0;
		for(uint64_t	i=0;	i<output;	i++){
			out[i]*=sca*alpha;
			if(out[i]>ma)	ma=out[i];
		}
		for(uint64_t	i=0;	i<output;	i++)	sum+=(out[i]=expf(out[i]-ma));
		for(uint64_t	i=0;	i<output;	i++)	out[i]/=sum;
		
	}
	float	backward(const	float	*inp,	const	uint64_t	target,	uint64_t	b=0){
		float	*gra=g(b),	*out=o(b);
		const	float	sca=1/sqrtf(input+1);	float	l=-logf(fmaxf(out[target],FLT_MIN));
		for(uint64_t	i=0;	i<output;	i++)	out[i]=(out[i]-(i==target))*sca*learning_rate;
		for(uint64_t	i=0;	i<=input;	i++){
			float	s=(i<input?inp[i]:1),	*p=w(i),	z=0;
			for(uint64_t	j=0;	j<output;	j++){	z+=p[j]*out[j];	p[j]-=s*out[j];	}
			if(i<input)	gra[i]=z;
		}
		return	l;
	}
};
