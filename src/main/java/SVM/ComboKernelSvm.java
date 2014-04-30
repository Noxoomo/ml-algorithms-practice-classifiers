package SVM;

import static java.lang.Math.exp;

/**
 * User: Vasily
 * Date: 04.12.13
 * Time: 20:28
 */
public class ComboKernelSvm extends SvmSMO {
    private double alpha = 0.01;

    public ComboKernelSvm(double[][] learn, double[] category) {
        super(learn, category);
    }

    public ComboKernelSvm(double[][] learn, double[] category, double C) {
        super(learn, category, C);
    }
    double kernel(double u[], double v[]) {
        double ker1 = 0;
        double ker2 = 0;
        for (int i=0;i<u.length;++i)  {
            ker1 += min(u[i],v[i]);
            ker2 += (u[i]-v[i])*(u[i]-v[i]);

        }

        return (1-alpha) * ker1 + alpha *exp(-ker2);
    }
    private double min(double a,double b) {
        if (a < b) return a;
        return b;
    }

}
