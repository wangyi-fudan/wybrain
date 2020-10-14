//data download:	http://yann.lecun.com/exdb/mnist/
#include	<sys/time.h>
#include	<iostream>
#include	"wybrain/wybrain.hpp"
#include	<vector>
#include	<zlib.h>
using	namespace	std;
const	unsigned	feature=784;
const	unsigned	hidden=128;
wl_normalize<feature>	xnorm;
wl_dropout<feature>	d1;
embed_dense<feature,hidden>	xembed;
fc_hidden<hidden,hidden>	h1;
fc_hidden<hidden,hidden>	h2;
lf_softmax<hidden,10>	out;

bool	load_image(const	char	*F,	vector<float>	&D,	unsigned	N){
	gzFile	in=gzopen(F,	"rb");
	if(in==Z_NULL)	return	false;
	unsigned	n;	gzread(in,	&n,	4);	gzread(in,	&n,	4);	gzread(in,	&n,	4);	gzread(in,	&n,	4);
	D.resize(N*feature);	vector<uint8_t>	temp(feature);
	for(size_t	i=0;	i<N;	i++){
		gzread(in,	temp.data(),	feature);
		for(size_t	j=0;	j<feature;	j++)	D[i*feature+j]=temp[j];
	}
	gzclose(in);	cerr<<F<<'\n';	return	true;
}

bool	load_label(const	char	*F,	vector<unsigned>	&D,	unsigned	N){
	gzFile	in=gzopen(F,	"rb");
	if(in==Z_NULL)	return	false;
	unsigned	n;	gzread(in,	&n,	4);	gzread(in,	&n,	4);	D.resize(N);	uint8_t	temp;
	for(size_t	i=0;	i<N;	i++){	gzread(in,	&temp,	1);	D[i]=temp;	}
	gzclose(in);	cerr<<F<<'\n';	return	true;
}

int	main(int	ac,	char	**av){
	cerr<<"***********************************\n";
	cerr<<"* MNIST                           *\n";
	cerr<<"* author: Yi Wang                 *\n";
	cerr<<"* email:  godspeed_china@yeah.net *\n";
	cerr<<"* date:   29/Oct/2019             *\n";
	cerr<<"***********************************\n";
	vector<float>	trainx,	testx;	
	vector<unsigned>	trainy,	testy;
	unsigned	trainn=60000,	testn=10000;	
	if(!load_image("train-images-idx3-ubyte.gz",	trainx,	trainn))	return	0;
	if(!load_image("t10k-images-idx3-ubyte.gz",	testx,	testn))	return	0;
	if(!load_label("train-labels-idx1-ubyte.gz",	trainy,	trainn))	return	0;
	if(!load_label("t10k-labels-idx1-ubyte.gz",	testy,	testn))	return	0;
	double	t0=0;	learning_rate=1;
	for(size_t	it=0;	learning_rate>0.001;	it++,learning_rate*=0.97){
		timeval	beg,	end;	gettimeofday(&beg,NULL);
		for(size_t	i=0;	i<trainn;	i++){
			size_t	ran=wyrand(&wybrain_seed)%trainn;
			xnorm.forward(trainx.data()+ran*feature,128,true);
			d1.forward(xnorm.o(0),0.5);
			xembed.forward(d1.o(0));
			h1.forward(xembed.o(0));
			h2.forward(h1.o(0));
			out.forward(h2.o(0));
			out.backward(h2.o(0),trainy[ran]);
			h2.backward(h1.o(0),out.g(0));
			h1.backward(xembed.o(0),h2.g(0));
			xembed.backward(d1.o(0),h1.g(0));
		}
		gettimeofday(&end,NULL);
		size_t	err=0;
		for(size_t	i=0;	i<testn;	i++){
			xnorm.forward(testx.data()+i*feature,128,true);
			d1.forward(xnorm.o(0),-0.5);
			xembed.forward(d1.o(0));
			h1.forward(xembed.o(0));
			h2.forward(h1.o(0));
			out.forward(h2.o(0));
			uint8_t	pre=0;
			for(size_t	j=0;	j<10;	j++)	if(out.o(0)[j]>out.o(0)[pre])	pre=j;
			err+=pre!=testy[i];
		}
		cerr.precision(3);	cerr.setf(ios::fixed);	t0+=(end.tv_sec-beg.tv_sec)+1e-6*(end.tv_usec-beg.tv_usec);
		cerr<<it<<'\t'<<"error="<<100.0*err/testn<<"%\teta="<<learning_rate<<"\ttime="<<t0<<"s\n";
	}
	return	0;
}
