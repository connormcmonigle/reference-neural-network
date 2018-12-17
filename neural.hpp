#include<iostream>
#include<functional>
#include<vector>
#include<array>
#include<algorithm>
#include<utility>
#include<cmath>

#define e 2.7182818

namespace neural{

	template<typename T, size_t I, size_t J> struct matrix{
		using M = std::array<std::array<T, J>, I>;
		M data;
		std::array<T, J>& operator[](size_t index){
			return data[index];
		}
		typename M::iterator begin(){
			return data.begin();
		}
		typename M::iterator end(){
			return data.end();
		}
		template<typename F> matrix<T, I, J>& apply(F&& f){
			for(auto& row : data){
				for(auto& x : row) x = f(x);
			}
			return (*this);
		}
	};

	template<typename T, size_t I, size_t J>
	matrix<T, I, J> operator+(matrix<T, I, J> a, matrix<T, I, J> b){
		for(size_t i = 0; i < I; ++i){
			for(size_t j = 0; j < J; ++j){
				a[i][j] += b[i][j];
			}
		}
		return a;
	}

	template<typename T, size_t I, size_t J, size_t K>
	matrix<T, I, K> operator*(matrix<T, I, J> a, matrix<T, J, K> b){
		matrix<T, I, K> c;
		for(size_t i = 0; i < I; ++i){
			for(size_t k = 0; k < K; ++k){
				for(size_t j = 0; j < J; ++j){
					c[i][k] += a[i][j] * b[j][k];
				}
			}
		}
		return c;
	}

	template<typename T> void printmat(T c){
		std::cout << "----------------------------" << std::endl;
		      for(auto& i : c){
		              std::cout << "{ ";
		              for(auto& j : i){
		                      std::cout << j << ", ";
		              }
		              std::cout << "}," << std::endl;
		      }
		std::cout << "----------------------------" << std::endl;
	}

	template<typename R, size_t I, size_t ... Ss>
	struct corpus{
		matrix<R, I, 1> derivatives;
		auto& set_learning_rate(R in){
			return *this;
		}
		  template<typename F, typename M> matrix<R, I, 1> run(F&& f, M&& in){
		  	return in;
		  }
		template<typename F, typename M> R erf(F&& f, M&& in, M&& e_out){
			auto iiter = in.begin();
			auto eiter = e_out.begin();
			auto diter = derivatives.begin();
			R error = R(0);
			for(; iiter != in.end() && eiter != e_out.end() && diter != derivatives.end(); ++iiter, ++eiter, ++diter){
				error += pow((*iiter)[0] - (*eiter)[0], R(2));
				(*diter)[0] = 2 * ((*iiter)[0] - (*eiter)[0]);
			}
			return error;
		}
		template<typename dF> matrix<R, I, 1> get_derivatives(dF&& df){
			return derivatives;
		}
	};

	template<typename R, size_t I, size_t J, size_t ... Ss>
	struct corpus<R, I, J, Ss...>{
		corpus<R, J, Ss...> rest;
		matrix<R, J, I> kernel;
		matrix<R, J, 1> bias;
		matrix<R, I, 1> last_input;
		matrix<R, J, 1> last_pre_output;
		R learning_rate = R(0);

		corpus<R, I, J, Ss...>& set_learning_rate(R in){
			learning_rate = in;
			rest.set_learning_rate(std::forward<R>(in));
			return *this;
		}

		template<typename F, typename M>
		auto run(F&& f, M&& in){
			return rest.run(std::forward<F>(f), 
			(kernel * in + bias).apply(std::forward<F>(f)));
		}

		template<typename F, typename M1, typename M2>
		R erf(F&& f, M1&& in, M2&& e_out){
			last_input = in;
			last_pre_output = kernel * in + bias;
			return rest.erf(std::forward<F>(f),
			last_pre_output.apply(std::forward<F>(f)), std::forward<M2>(e_out));
		}

		template<typename dF>
		matrix<R, I, 1> get_derivatives(dF&& df){
			matrix<R, J, 1> dE = rest.get_derivatives(std::forward<dF>(df));
			matrix<R, I, 1> d;
			for(int j = 0; j < J; ++j){
				for(int i = 0; i < I; ++i){
					d[i][0] += dE[j][0] * df(last_pre_output[j][0]) * kernel[j][i];
				}
			}
			for(int j = 0; j < J; ++j){
				for(int i = 0; i < I; ++i){
					kernel[j][i] -= dE[j][0] * df(last_pre_output[j][0]) * last_input[i][0] * learning_rate;
				}
			}
			for(int j = 0; j < J; ++j){
				bias[j][0] -= dE[j][0] * df(last_pre_output[j][0]);
			}
			return d;
		}

		corpus(){
			for(auto& row : kernel){
				for(auto& x : row){
					x = (R(rand() % (1 << 24)) / R(1 << 23)) - R(1);
				}
			}
			for(auto& x : bias){
				x[0] = (R(rand() % (1 << 24)) / R(1 << 23)) - R(1);
			}
		}
	};

}
