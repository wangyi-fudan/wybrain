#include	"common.hpp"
/*
template<uint64_t	input,	uint64_t	batch=1>
struct	wl_softmax{
	matrix<batch,input>	g;
	matrix<batch,input>	o;
	void	forward(const	float	*inp,	uint64_t	b=0){
		float	*out=o(b);
		float	ma=-FLT_MAX,	sum=0;
		for(uint64_t	i=0;	i<input;	i++)	if(inp[i]>ma)	ma=inp[i];
		for(uint64_t	i=0;	i<input;	i++)	sum+=(out[i]=expf(inp[i]-ma));
		for(uint64_t	i=0;	i<input;	i++)	out[i]/=sum;
	}
	void	backward(const	float	*bac,	uint64_t	b=0){
		float	*gra=g(b),	*out=o(b);
		for(uint64_t	i=0;	i<input;	i++)	gra[i]=out[i]*(1-out[i])*bac[i];
	}
};
*/
template<uint64_t	input,	uint64_t	batch=1>
struct	wl_standardize{
	const	double	decay=0.999999;
	matrix<input,3,double>	w;
	wl_standardize(){	for(uint64_t	i=0;	i<input;	i++){	w(i)[0]=w(i)[1]=w(i)[2]=0;	}	}
	matrix<batch,input>	g;
	matrix<batch,input>	o;
	void	forward(const	float	*inp,	bool	update,	uint64_t	b=0){
		float	*out=o(b);
		for(uint64_t	i=0;	i<input;	i++){
			if(update){
				w(i)[0]=w(i)[0]*decay+1;
				w(i)[1]=w(i)[1]*decay+inp[i];
				w(i)[2]=w(i)[2]*decay+inp[i]*inp[i];
			}
			double	m=w(i)[1]/w(i)[0],	d=(w(i)[2]+1)/w(i)[0]-m*m;
			out[i]=(inp[i]-m)/sqrt(d);
		}
	}
	void	backward(float	*bac,	uint64_t	b=0){
		float	*gra=g(b);
		for(uint64_t	i=0;	i<input;	i++){
			double	m=w(i)[1]/w(i)[0],	d=(w(i)[2]+1)/w(i)[0]-m*m;
			gra[i]=bac[i]/sqrt(d);
		}
	}
	void	original(const	float	*inp,	uint64_t	b=0){
		float	*gra=g(b);
		for(uint64_t	i=0;	i<input;	i++){
			double	m=w(i)[1]/w(i)[0],	d=(w(i)[2]+1)/w(i)[0]-m*m;
			gra[i]=inp[i]*sqrt(d)+m;
		}
	}
};

template<uint64_t	input,	uint64_t	batch=1>
struct	wl_normalize{
	const	double	decay=0.999999;
	matrix<input,2,double>	w;
	wl_normalize(){	for(uint64_t	i=0;	i<input;	i++){	w(i)[0]=w(i)[1]=0;	}	}
	matrix<batch,input>	g;
	matrix<batch,input>	o;
	void	forward(const	float	*inp,	float	stdev,	bool	update,	uint64_t	b=0){
		float	*out=o(b);
		for(uint64_t	i=0;	i<input;	i++){
			if(update){
				w(i)[0]=w(i)[0]*decay+1;
				w(i)[1]=w(i)[1]*decay+inp[i];
			}
			double	m=w(i)[1]/w(i)[0];
			out[i]=(inp[i]-m)/stdev;
		}
	}
	void	backward(float	*bac,	float	stdev,	uint64_t	b=0){
		float	*gra=g(b);
		for(uint64_t	i=0;	i<input;	i++)	gra[i]=bac[i]/stdev;
	}
	void	original(const	float	*inp,	float	stdev,	uint64_t	b=0){
		float	*gra=g(b);
		for(uint64_t	i=0;	i<input;	i++){
			double	m=w(i)[1]/w(i)[0];
			gra[i]=inp[i]*stdev+m;
		}
	}
};

template<uint64_t	input,	uint64_t	batch=1>
struct	wl_dropout{
	matrix<batch,input>	g;
	matrix<batch,input>	o;
	void	forward(const	float	*inp,	double	drop,	uint64_t	b=0){
		float	*out=o(b);
		if(drop>0)	for(uint64_t	i=0;	i<input;	i++)	out[i]=(wy2u01(wyrand(&wybrain_seed))>drop)*inp[i];
		else	if(drop<0)	for(uint64_t	i=0;	i<input;	i++)	out[i]=(1+drop)*inp[i];
		else	memcpy(out,	inp,	input*sizeof(float));
	}
	void	backward(float	*bac,	uint64_t	b=0){
		float	*gra=g(b),	*out=o(b);
		for(uint64_t	i=0;	i<input;	i++)	gra[i]=bac[i]*(out[i]!=0);
	}
};

template<uint64_t	input,	uint64_t	batch=1>
struct	wl_noise{
	matrix<batch,input>	g;
	matrix<batch,input>	o;
	void	forward(const	float	*inp,	double	noise,	uint64_t	b=0){
		float	*out=o(b);
		if(noise>0)	for(uint64_t	i=0;	i<input;	i++)	out[i]=noise*wy2gau(wyrand(&wybrain_seed))+inp[i];
		else	memcpy(out,	inp,	input*sizeof(float));
	}
	void	backward(float	*bac,	uint64_t	b=0){
		float	*gra=g(b);
		memcpy(gra,	bac,	input*sizeof(float));
	}
};

template<uint64_t	input,	uint64_t	batch=1>
struct	wl_dot_product{
	matrix<batch,input>	g0,g1;
	matrix<batch,1>	o;
	void	forward(const	float	*inp0,	const	float	*inp1,	uint64_t	b=0){
		float	*out=o(b);	*out=0;
		for(uint64_t	i=0;	i<input;	i++)	*out+=inp0[i]*inp1[i];
		*out/=sqrt(input);
	}
	void	backward(const	float	*inp0,  const	float	*inp1,	float	*bac,	uint64_t	b=0){
		float	*gra0=g0(b),	*gra1=g1(b),	s=*bac/sqrt(input);
		for(uint64_t	i=0;	i<input;	i++){	gra0[i]=s*inp1[i];	gra1[i]=s*inp0[i];	}
	}
};

template<uint64_t	width,	uint64_t	height,	uint64_t	batch=1>
struct	wl_2d_random_patch{
	matrix<batch,height*width>	o;
	void	forward(const	float	*inp,	uint64_t	w,	uint64_t	h,	uint64_t	b=0){
		float	*out=o(b);
		memset(out,	0,	height*width*sizeof(float));
		uint64_t	x=wy2u0k(wyrand(&wybrain_seed),w-width+1);
		uint64_t	y=wy2u0k(wyrand(&wybrain_seed),h-height+1);
		for(uint64_t	i=0;	i<height;	i++)	for(uint64_t	j=0;	j<width;	j++)	out[i*width+j]=inp[(i+y)*w+j+x];
	}
};

template<int	width,	int	height,	uint64_t	batch=1>
struct	wl_2d_random_rotated_patch{
	matrix<batch,height*width>	o;
	void	forward(const	float	*inp,	int	w,	int	h,	float	theta,	uint64_t	b=0){
		float	*out=o(b);
		memset(out,	0,	height*width*sizeof(float));
		float	x=wy2u01(wyrand(&wybrain_seed))*(w-width);
		float	y=wy2u01(wyrand(&wybrain_seed))*(h-height);
		float	cx=0.5f*width,	cy=0.5f*height,	t=(wy2u01(wyrand(&wybrain_seed))-0.5)*theta,	cost=cosf(t),	sint=sinf(t);
		for(int	i=0;	i<height;	i++)	for(int	j=0;	j<width;	j++){
			int	tx=(j-cx)*cost+(i-cy)*sint+x+cx,	ty=-(j-cx)*sint+(i-cy)*cost+y+cy;
			if(tx>=0&&tx<w&&ty>=0&&ty<h)	out[i*width+j]=inp[ty*w+tx];
		}
	}
};
