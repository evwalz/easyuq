#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cmath> 
#include <math.h>
#include <vector>


#include "thirdparty/boost/boost/math/quadrature/gauss_kronrod.hpp"
#include "thirdparty/boost/boost/math/distributions/normal.hpp"
#include "thirdparty/boost/boost/math/distributions/students_t.hpp"


#include <iostream>
using boost::math::quadrature::gauss_kronrod;
using boost::math::normal;
using boost::math::students_t;

namespace py = pybind11;



double crps_norm_lims(py::array_t<double>& Y, py::array_t<double>& M, py::array_t<double>& W, py::array_t<int>& I, double h, double low, double up) {

    py::buffer_info bufY = Y.request();
    double* ptrY = (double*)bufY.ptr;
    py::ssize_t n = bufY.shape[0];

    py::buffer_info bufM = M.request();
    double* ptrM = (double*)bufM.ptr;
    
    py::buffer_info bufW = W.request();
    double* ptrW = (double*)bufW.ptr;

    py::buffer_info bufI = I.request();
    int* ptrI = (int*)bufI.ptr;
 
    double out = 0;
        
    double inf = std::numeric_limits<double>::infinity();

    const double mean = 0;
    const double stddev = 1;
    
    for (int i = 0; i < n; i++){
        double yval = ptrY[i];
        int s0 = ptrI[i];
        int s1 = ptrI[i+1];
        int end = s1 - s0;
        std::vector<double> w0(end);
        std::vector<double> m0(end);
        for(int k = 0; k < end; k++){
            w0[k] = ptrW[k+s0];
            m0[k] = ptrM[k+s0];
        }        
   
        normal dist(mean, stddev);
       
        auto f1 = [&dist, &h, &m0, &w0, &end](double t) 
            { 
		double sum = 0;
 		for(int j = 0; j < end; j++){
		    double z = (t-m0[j])/h;
		    sum = sum + w0[j]*cdf(dist, z);
		}
                return sum*sum;		
            };
        auto f2 = [&dist, &h, &m0, &w0, &end](double t) 
            { 
	        double sum = 0;
 		for(int j = 0; j < end; j++){
		    double z = (t-m0[j])/h;
		    sum = sum + w0[j]*cdf(dist, z);
		}

		return (1-sum)*(1-sum); 
	    };
        double error;
        double I1 = gauss_kronrod<double, 15>::integrate(f1, low, yval, 0, 0, &error);
        double I2 = gauss_kronrod<double, 15>::integrate(f2, yval, up, 0, 0, &error);

        out = out + I1 + I2; 
    
    }
    return out / n;
}

double crps_t_lims(py::array_t<double>& Y, py::array_t<double>& M, py::array_t<double>& W, py::array_t<int>& I, double h, double df, double low, double up) {

    py::buffer_info bufY = Y.request();
    double* ptrY = (double*)bufY.ptr;
    py::ssize_t n = bufY.shape[0];

    py::buffer_info bufM = M.request();
    double* ptrM = (double*)bufM.ptr;
    
    py::buffer_info bufW = W.request();
    double* ptrW = (double*)bufW.ptr;

    py::buffer_info bufI = I.request();
    int* ptrI = (int*)bufI.ptr;
 
    double out = 0;
        
    double inf = std::numeric_limits<double>::infinity();
    
    for (int i = 0; i < n; i++){
        double yval = ptrY[i];
        int s0 = ptrI[i];
        int s1 = ptrI[i+1];
        int end = s1 - s0;
        std::vector<double> w0(end);
        std::vector<double> m0(end);
        for(int k = 0; k < end; k++){
            w0[k] = ptrW[k+s0];
            m0[k] = ptrM[k+s0];
        }        
   
        students_t dist(df);
       
        auto f1 = [&dist, &h, &m0, &w0, &end](double t) 
            { 
		double sum = 0;
 		for(int j = 0; j < end; j++){
		    double z = (t-m0[j])/h;
		    sum = sum + w0[j]*cdf(dist, z);
		}
                return sum*sum;		
            };
        auto f2 = [&dist, &h, &m0, &w0, &end](double t) 
            { 
	        double sum = 0;
 		for(int j = 0; j < end; j++){
		    double z = (t-m0[j])/h;
		    sum = sum + w0[j]*cdf(dist, z);
		}

		return (1-sum)*(1-sum); 
	    };
        double error;
        double I1 = gauss_kronrod<double, 15>::integrate(f1, low, yval, 0, 0, &error);
        double I2 = gauss_kronrod<double, 15>::integrate(f2, yval, up, 0, 0, &error);

        out = out + I1 + I2; 
    
    }
    return out / n;
}

//* compute integral from -1*infinity to 0:*//
double crps_norm_censored(py::array_t<double>& Y, py::array_t<double>& M, py::array_t<double>& W, py::array_t<int>& I, double h, double low, double up) {

    py::buffer_info bufY = Y.request();
    double* ptrY = (double*)bufY.ptr;
    py::ssize_t n = bufY.shape[0];

    py::buffer_info bufM = M.request();
    double* ptrM = (double*)bufM.ptr;
    
    py::buffer_info bufW = W.request();
    double* ptrW = (double*)bufW.ptr;

    py::buffer_info bufI = I.request();
    int* ptrI = (int*)bufI.ptr;
 
    double out = 0;
        
    double inf = std::numeric_limits<double>::infinity();

    const double mean = 0;
    const double stddev = 1;
    
    for (int i = 0; i < n; i++){
        double yval = ptrY[i];
        int s0 = ptrI[i];
        int s1 = ptrI[i+1];
        int end = s1 - s0;
        std::vector<double> w0(end);
        std::vector<double> m0(end);
        for(int k = 0; k < end; k++){
            w0[k] = ptrW[k+s0];
            m0[k] = ptrM[k+s0];
        }        
   
        normal dist(mean, stddev);
       
        auto f1 = [&dist, &h, &m0, &w0, &end](double t) 
            { 
		double sum = 0;
 		for(int j = 0; j < end; j++){
		    double z = (t-m0[j])/h;
		    sum = sum + w0[j]*cdf(dist, z);
		}
                return sum*sum;		
            };

        double error;
        double I1 = gauss_kronrod<double, 15>::integrate(f1, low, up, 0, 0, &error);

        out = out + I1; 
    
    }
    return out / n;
}


double crps_t_censored(py::array_t<double>& Y, py::array_t<double>& M, py::array_t<double>& W, py::array_t<int>& I, double h, double df, double low, double up) {

    py::buffer_info bufY = Y.request();
    double* ptrY = (double*)bufY.ptr;
    py::ssize_t n = bufY.shape[0];

    py::buffer_info bufM = M.request();
    double* ptrM = (double*)bufM.ptr;
    
    py::buffer_info bufW = W.request();
    double* ptrW = (double*)bufW.ptr;

    py::buffer_info bufI = I.request();
    int* ptrI = (int*)bufI.ptr;
 
    double out = 0;
        
    
    for (int i = 0; i < n; i++){
        double yval = ptrY[i];
        int s0 = ptrI[i];
        int s1 = ptrI[i+1];
        int end = s1 - s0;
        std::vector<double> w0(end);
        std::vector<double> m0(end);
        for(int k = 0; k < end; k++){
            w0[k] = ptrW[k+s0];
            m0[k] = ptrM[k+s0];
        }        
   
        students_t dist(df);
       
        auto f1 = [&dist, &h, &m0, &w0, &end](double t) 
            { 
		double sum = 0;
 		for(int j = 0; j < end; j++){
		    double z = (t-m0[j])/h;
		    sum = sum + w0[j]*cdf(dist, z);
		}
                return sum*sum;		
            };
        auto f2 = [&dist, &h, &m0, &w0, &end](double t) 
            { 
	        double sum = 0;
 		for(int j = 0; j < end; j++){
		    double z = (t-m0[j])/h;
		    sum = sum + w0[j]*cdf(dist, z);
		}

		return (1-sum)*(1-sum); 
	    };

        double error;
        double I2 = gauss_kronrod<double, 15>::integrate(f2, yval, up, 0, 0, &error);
        out = out + I2;
        
    }
    return out / n;
}


double crps_int_below0(py::array_t<double>& Y, py::array_t<double>& M, py::array_t<double>& W, py::array_t<int>& I, double h, double df, double low, double up) {

    py::buffer_info bufY = Y.request();
    double* ptrY = (double*)bufY.ptr;
    py::ssize_t n = bufY.shape[0];

    py::buffer_info bufM = M.request();
    double* ptrM = (double*)bufM.ptr;
    
    py::buffer_info bufW = W.request();
    double* ptrW = (double*)bufW.ptr;

    py::buffer_info bufI = I.request();
    int* ptrI = (int*)bufI.ptr;
 
    double out = 0;
        
    
    for (int i = 0; i < n; i++){
        double yval = ptrY[i];
        int s0 = ptrI[i];
        int s1 = ptrI[i+1];
        int end = s1 - s0;
        std::vector<double> w0(end);
        std::vector<double> m0(end);
        for(int k = 0; k < end; k++){
            w0[k] = ptrW[k+s0];
            m0[k] = ptrM[k+s0];
        }        
   
        students_t dist(df);
       
        auto f1 = [&dist, &h, &m0, &w0, &end](double t) 
            { 
		double sum = 0;
 		for(int j = 0; j < end; j++){
		    double z = (t-m0[j])/h;
		    sum = sum + w0[j]*cdf(dist, z);
		}
                return sum*sum;		
            };

        double error;
        double I2 = gauss_kronrod<double, 15>::integrate(f1, low, up, 0, 0, &error);
        out = out + I2;
        
    }
    return out / n;
}




double crps_norm_hvec(py::array_t<double>& Y, py::array_t<double>& M, py::array_t<double>& W, py::array_t<int>& I, py::array_t<double>& h, double low, double up) {

    py::buffer_info bufY = Y.request();
    double* ptrY = (double*)bufY.ptr;
    py::ssize_t n = bufY.shape[0];

    py::buffer_info bufM = M.request();
    double* ptrM = (double*)bufM.ptr;
    
    py::buffer_info bufW = W.request();
    double* ptrW = (double*)bufW.ptr;

    py::buffer_info bufI = I.request();
    int* ptrI = (int*)bufI.ptr;

    py::buffer_info bufh = h.request();
    double* ptrh = (double*)bufh.ptr; 
    
    double out = 0;
        
    double inf = std::numeric_limits<double>::infinity();

    const double mean = 0;
    const double stddev = 1;
    
    for (int i = 0; i < n; i++){
        double h = ptrh[i];
        double yval = ptrY[i];
        int s0 = ptrI[i];
        int s1 = ptrI[i+1];
        int end = s1 - s0;
        std::vector<double> w0(end);
        std::vector<double> m0(end);
        for(int k = 0; k < end; k++){
            w0[k] = ptrW[k+s0];
            m0[k] = ptrM[k+s0];
        }        
   
        normal dist(mean, stddev);
       
        auto f1 = [&dist, &h, &m0, &w0, &end](double t) 
            { 
		double sum = 0;
 		for(int j = 0; j < end; j++){
		    double z = (t-m0[j])/h;
		    sum = sum + w0[j]*cdf(dist, z);
		}
                return sum*sum;		
            };
        auto f2 = [&dist, &h, &m0, &w0, &end](double t) 
            { 
	        double sum = 0;
 		for(int j = 0; j < end; j++){
		    double z = (t-m0[j])/h;
		    sum = sum + w0[j]*cdf(dist, z);
		}

		return (1-sum)*(1-sum); 
	    };
        double error;
        double I1 = gauss_kronrod<double, 15>::integrate(f1, low, yval, 0, 0, &error);
        double I2 = gauss_kronrod<double, 15>::integrate(f2, yval, up, 0, 0, &error);

        out = out + I1 + I2; 
    
    }
    return out / n;
}


double crps_norm_hvec_below0(py::array_t<double>& Y, py::array_t<double>& M, py::array_t<double>& W, py::array_t<int>& I, py::array_t<double>& h, double low, double up) {

    py::buffer_info bufY = Y.request();
    double* ptrY = (double*)bufY.ptr;
    py::ssize_t n = bufY.shape[0];

    py::buffer_info bufM = M.request();
    double* ptrM = (double*)bufM.ptr;
    
    py::buffer_info bufW = W.request();
    double* ptrW = (double*)bufW.ptr;

    py::buffer_info bufI = I.request();
    int* ptrI = (int*)bufI.ptr;

    py::buffer_info bufh = h.request();
    double* ptrh = (double*)bufh.ptr; 
    
    double out = 0;
        
    double inf = std::numeric_limits<double>::infinity();

    const double mean = 0;
    const double stddev = 1;
    
    for (int i = 0; i < n; i++){
        double h = ptrh[i];
        double yval = ptrY[i];
        int s0 = ptrI[i];
        int s1 = ptrI[i+1];
        int end = s1 - s0;
        std::vector<double> w0(end);
        std::vector<double> m0(end);
        for(int k = 0; k < end; k++){
            w0[k] = ptrW[k+s0];
            m0[k] = ptrM[k+s0];
        }        
   
        normal dist(mean, stddev);
       
        auto f1 = [&dist, &h, &m0, &w0, &end](double t) 
            { 
		double sum = 0;
 		for(int j = 0; j < end; j++){
		    double z = (t-m0[j])/h;
		    sum = sum + w0[j]*cdf(dist, z);
		}
                return sum*sum;		
            };
        double error;
        double I1 = gauss_kronrod<double, 15>::integrate(f1, low, up, 0, 0, &error);

        out = out + I1; 
    
    }
    return out / n;
}


double crps_norm_censored_hvec(py::array_t<double>& Y, py::array_t<double>& M, py::array_t<double>& W, py::array_t<int>& I, py::array_t<double>& h, double low, double up) {

    py::buffer_info bufY = Y.request();
    double* ptrY = (double*)bufY.ptr;
    py::ssize_t n = bufY.shape[0];

    py::buffer_info bufM = M.request();
    double* ptrM = (double*)bufM.ptr;
    
    py::buffer_info bufW = W.request();
    double* ptrW = (double*)bufW.ptr;

    py::buffer_info bufI = I.request();
    int* ptrI = (int*)bufI.ptr;

    py::buffer_info bufh = h.request();
    double* ptrh = (double*)bufh.ptr; 
    
    double out = 0;
        
    double inf = std::numeric_limits<double>::infinity();

    const double mean = 0;
    const double stddev = 1;
    
    for (int i = 0; i < n; i++){
        double h = ptrh[i];
        double yval = ptrY[i];
        int s0 = ptrI[i];
        int s1 = ptrI[i+1];
        int end = s1 - s0;
        std::vector<double> w0(end);
        std::vector<double> m0(end);
        for(int k = 0; k < end; k++){
            w0[k] = ptrW[k+s0];
            m0[k] = ptrM[k+s0];
        }        
   
        normal dist(mean, stddev);
       
        auto f2 = [&dist, &h, &m0, &w0, &end](double t) 
            { 
	        double sum = 0;
 		for(int j = 0; j < end; j++){
		    double z = (t-m0[j])/h;
		    sum = sum + w0[j]*cdf(dist, z);
		}

		return (1-sum)*(1-sum); 
	    };
        
        double error;
        double I1 = gauss_kronrod<double, 15>::integrate(f2, yval, up, 0, 0, &error);

        out = out + I1; 
    
    }
    return out / n;
}


PYBIND11_MODULE(crps_lims, m) {
    m.doc() = "crps computation of mixture distributions"; // optional module docstring

    m.def("crps_norm_lims", &crps_norm_lims, "Computes CRPS for mixture of normal distributions for observations y");
    m.def("crps_t_lims", &crps_t_lims, "Computes CRPS for mixture of t-distribution for observations y");
    m.def("crps_norm_censored", &crps_norm_censored, "Computes CRPS for censored mixture of normal distributions for observations y");
    m.def("crps_t_censored", &crps_t_censored, "Computes CRPS for censored mixture of t-distribution for observations y");
    m.def("crps_int_below0", &crps_int_below0, "Computes CRPS-Integral for mixture of t-distribution from -Inf to 0");
    m.def("crps_norm_hvec", &crps_norm_hvec, "Computes CRPS for mixture of normal distribution with varying bandwidth values for observations y");
    m.def("crps_norm_censored_hvec", &crps_norm_censored_hvec, "Computes CRPS for censored mixture of normal distribution with varying bandwidth values for observations y");
    m.def("crps_norm_hvec_below0", &crps_norm_hvec_below0, "Computes CRPS_Integral for censored mixture of normal distribution with varying bandwidth values from -Inf to y");
}

