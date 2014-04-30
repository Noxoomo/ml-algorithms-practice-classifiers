package SVM;

/**
 * User: Vasily
 * Date: 08.12.13
 * Time: 17:11
 */
public class MinKernel extends SvmSMO {
    public MinKernel(double[][] learn, double[] category, double C) {
        super(learn, category, C);
    }

    public MinKernel(double[][] learn, double[] category) {
        super(learn, category);
    }

    double kernel(double u[], double v[]) {
        if (u.length != v.length) {
            throw new RuntimeException();
        }
        double result = 0;
        for (int i = 0; i < u.length; ++i) {
            result += min(u[i],v[i]);
        }
        return result;
    }
    private double min(double a,double b) {
        if (a < b) return a;
        return b;
    }

}
