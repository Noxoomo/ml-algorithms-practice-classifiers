package SVM;

import static java.lang.Math.exp;

/**
 * User: Vasily
 * Date: 30.11.13
 * Time: 22:50
 */
public class RadialBasisSvm extends SvmSMO {
    private  double gamma = 1.0 / 42;

    public RadialBasisSvm(double[][] learn, double[] category, double C) {
        super(learn, category, C);
    }
    public void  setGamma(double gamma) {
        this.gamma = gamma;
    }

    public RadialBasisSvm(double[][] learn, double[] category) {
        super(learn, category);
    }
    @Override
    double kernel(double u[], double v[]) {
        double distance = 0;
        for (int i=0;i<u.length;++i)
            distance += (u[i]-v[i]) * (u[i]-v[i]);
        return exp(-gamma*distance);
    }

}
