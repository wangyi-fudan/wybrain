#include	"common.hpp"
template<unsigned	input,	unsigned	batch=1>
struct	wl_softmax{
	matrix<batch,input>	g;
	matrix<batch,input>	o;
	void	forward(const	float	*inp,	unsigned	b=0){
		float	*out=o(b);
		float	ma=-FLT_MAX,	sum=0;
		for(unsigned	i=0;	i<input;	i++)	if(inp[i]>ma)	ma=inp[i];
		for(unsigned	i=0;	i<input;	i++)	sum+=(out[i]=expf(inp[i]-ma));
		for(unsigned	i=0;	i<input;	i++)	out[i]/=sum;
	}
	void	backward(const	float	*gradient,	unsigned	b=0){
		float	*gra=g(b),	*out=o(b);
		for(unsigned	i=0;	i<input;	i++)	gra[i]=out[i]*(1-out[i])*gradient[i];
	}
};

template<unsigned	input,	unsigned	batch=1>
struct	wl_standardize{
	const	double	decay=0.999999;
	matrix<input,3,double>	w;
	wl_standardize(){	for(unsigned	i=0;	i<input;	i++){	w(i)[0]=1;	w(i)[1]=0;	w(i)[2]=1;	}	}
	matrix<batch,input>	g;
	matrix<batch,input>	o;
	void	forward(const	float	*inp,	unsigned	b=0){
		float	*out=o(b);
		for(unsigned	i=0;	i<input;	i++){
			w(i)[0]=w(i)[0]*decay+1;
			w(i)[1]=w(i)[1]*decay+inp[i];
			w(i)[2]=w(i)[2]*decay+inp[i]*inp[i];
			double	m=w(i)[1]/w(i)[0],	d=w(i)[2]/w(i)[0]-m*m;
			out[i]=d>0?(inp[i]-m)/sqrt(d):0;
		}
	}
	void	backward(float	*gradient,	unsigned	b=0){
		float	*gra=g(b);
		for(unsigned	i=0;	i<input;	i++){
			double	m=w(i)[1]/w(i)[0],	d=w(i)[2]/w(i)[0]-m*m;
			gra[i]=gradient[i]*(d>0?sqrt(d):0);
		}
	}
	void	original(const	float	*inp,	unsigned	b=0){
		float	*gra=g(b);
		for(unsigned	i=0;	i<input;	i++){
			double	m=w(i)[1]/w(i)[0],	d=w(i)[2]/w(i)[0]-m*m;
			gra[i]=d>0?inp[i]*sqrt(d)+m:m;
		}
	}
};

template<unsigned	input,	unsigned	batch=1>
struct	wl_dropout{
	matrix<batch,input>	g;
	matrix<batch,input>	o;
	void	forward(const	float	*inp,	double	drop,	unsigned	b=0){
		float	*out=o(b);
		if(drop>0)	for(unsigned	i=0;	i<input;	i++)	out[i]=(wy2u01(wyrand(&wybrain_seed))>drop)*inp[i];
		else	if(drop<0)	for(unsigned	i=0;	i<input;	i++)	out[i]=(1-drop)*inp[i];
		else	memcpy(out,	inp,	input*sizeof(float));
	}
	void	backward(float	*gradient,	unsigned	b=0){
		float	*gra=g(b),	*out=o(b);
		for(unsigned	i=0;	i<input;	i++)	gra[i]=gradient[i]*(out[i]!=0);
	}
};

template<unsigned	input,	unsigned	batch=1>
struct	wl_noise{
	matrix<batch,input>	g;
	matrix<batch,input>	o;
	void	forward(const	float	*inp,	double	noise,	unsigned	b=0){
		float	*out=o(b);
		if(noise>0)	for(unsigned	i=0;	i<input;	i++)	out[i]=noise*wy2gau(wyrand(&wybrain_seed))+inp[i];
		else	memcpy(out,	inp,	input*sizeof(float));
	}
	void	backward(float	*gradient,	unsigned	b=0){
		float	*gra=g(b);
		memcpy(gra,	gradient,	input*sizeof(float));
	}
};

template<unsigned	input,	unsigned	batch=1>
struct	wl_dot_product{
	matrix<batch,input>	g0,g1;
	matrix<batch,1>	o;
	void	forward(const	float	*inp0,	const	float	*inp1,	unsigned	b=0){
		float	*out=o(b);	*out=0;
		for(unsigned	i=0;	i<input;	i++)	*out+=inp0[i]*inp1[i];
		*out/=sqrt(input);
	}
	void	backward(const	float	*inp0,  const	float	*inp1,	float	*gradient,	unsigned	b=0){
		float	*gra0=g0(b),	*gra1=g1(b),	s=*gradient/sqrt(input);
		for(unsigned	i=0;	i<input;	i++){	gra0[i]=s*inp1[i];	gra1[i]=s*inp0[i];	}
	}
};

