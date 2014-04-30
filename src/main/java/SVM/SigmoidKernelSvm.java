package SVM;

import static java.lang.Math.tanh;

/**
 * User: Vasily
 * Date: 04.12.13
 * Time: 20:41
 */
public class SigmoidKernelSvm extends SvmSMO {
    private double gamma = 1.0 / 42;
    private double coef = 0;
    public void setGamma(double gamma) {
        this.gamma = gamma;
    }
    public void setCoef(double coef) {
        this.coef = coef;
    }

    public SigmoidKernelSvm(double[][] learn, double[] category, double C) {
        super(learn, category, C);
    }

    public SigmoidKernelSvm(double[][] learn, double[] category) {
        super(learn, category);
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
        return tanh(gamma * result + coef);
    }
}
