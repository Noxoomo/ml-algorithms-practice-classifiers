package RandomForest;

import java.util.Random;

/**
 * User: Vasily
 * Date: 25.04.14
 * Time: 23:03
 */
public class Utils {
    static Random random = new Random();

    public static int[] argsort(int[] data) {
        int[] args = new int[data.length];
        for (int i = 0; i < data.length; ++i)
            args[i] = i;


        return data;
    }

}
