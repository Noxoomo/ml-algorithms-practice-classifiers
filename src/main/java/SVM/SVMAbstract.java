package SVM;

/**
 * User: Vasily
 * Date: 30.11.13
 * Time: 23:01
 */
public abstract class SVMAbstract {
    abstract double kernel(double u[], double v[]);

    abstract double[] predict(double observation[][]);
}
