package SVM;

import static java.lang.Math.pow;

/**
 * User: Vasily
 * Date: 07.12.13
 * Time: 15:35
 */
public class PolynomialSvm extends SvmSMO {
    private double gamma = 0.001;
    private double coef = 1;
    private double degree  = 33;

    public PolynomialSvm(double[][] learn, double[] category, double C) {
        super(learn, category, C);
    }

    public PolynomialSvm(double[][] learn, double[] category) {
        super(learn, category);
    }

    public void setGamma(double gamma) {
        this.gamma = gamma;
    }
    public void setDegree(double degree) {
        this.degree = degree;
    }
    public  void setCoef(double coef) {
        this.coef = coef;
    }

    @Override
    double kernel(double[] u, double[] v) {
        if (u.length != v.length) {
            throw new RuntimeException();
        }
        double result = 0;
        for (int i = 0; i < u.length; ++i) {
            result += u[i] * v[i];
        }
        return pow(gamma*result+coef,degree);
    }

}
