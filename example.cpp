#include<iostream>
#include<functional>
#include<vector>
#include<array>
#include<algorithm>
#include<utility>
#include<cmath>
#include "neural.hpp"

using namespace std;

template<typename F, typename dF, typename M, typename C, typename R, size_t I>
void sgd_optimize(C&& corp, R&& lr, F&& f, dF&& df, vector<pair<neural::matrix<R, I, 1>, M>> data, size_t times){
	corp.set_learning_rate(lr);
		for(size_t n = 0; n < times; ++n){
			R avg = R(0);
        	for(auto& elem : data){
                avg += corp.erf(forward<F>(f), get<0>(elem), get<1>(elem));
        		corp.get_derivatives(forward<dF>(df));
        	}
			cout << avg / R(data.size()) << endl;
		}
}

int main(){
	neural::corpus<double, 4, 3, 1> c;
	auto f = [](double x){
		return double(1) / (double(1) + pow(e, -x));
	};
	auto df = [&f](double x){
		return f(x) * (1 - f(x));
	};
	auto random = [](auto in){return double(rand() % 10000) / double(10000);};
	vector<pair<neural::matrix<double, 4, 1>, neural::matrix<double, 1, 1>>> data(4);
	for(auto& elem : data){
		get<0>(elem).apply(random);
		get<1>(elem).apply(random);
	}
	sgd_optimize(c, 0.03, f, df, data, 1000000);
}
