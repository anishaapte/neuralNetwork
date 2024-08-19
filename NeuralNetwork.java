import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Date;
import java.util.List;
import java.util.Random;

import Jama.Matrix;

public class NeuralNetwork {

    public List<Integer> labels;
    public int[] testLabels;
    public List<double[]> images;
    public List<double[]> testImages;
    public  final double ALPHA = .00001;
    
    static final int HID = 28;
    static final int M = 784;

    double[][] W1 = new double[M][HID];
    double[][] B1 = new double[HID][1];
    double[][] W2 = new double[HID][1];
    double B2;


    public static void main(String[] args) {
        // TODO Auto-generated method stub
        NeuralNetwork nn = new NeuralNetwork();
        nn.processText("/Users/anishaapte/Desktop/AI540/mnist_test.csv");
        nn.processTest("/Users/anishaapte/Desktop/AI540/regTest.txt");
        
        nn.initWeights();
        nn.train();
        nn.test();

    }
    public void printArr(String s) {
        System.out.println(s.substring(1, s.length() - 1));
    }
    
    public NeuralNetwork() {
        
    }
    
    public void processText(String filename) {
        labels = new ArrayList<Integer>();
        images = new ArrayList<double[]>();
        
        try {
            FileReader f = new FileReader(filename);
            BufferedReader b = new BufferedReader(f);
            String current = b.readLine();
            int count1 = 0;
            int count0 = 0;
            
            while (current != null) {
                String[] tokens = current.split(",");
                
                //setting the label
                int label = Integer.parseInt(tokens[0]);   
                if (label == 2) {
                    label = 1;
                    count1++;
                }
                else {
                    label = 0;
                    count0++;
                }
                
                //adding remaining columns 
                double[] vals = new double[tokens.length - 1];               
                for (int i = 0; i < tokens.length - 1; i++) {
                    vals[i] = Double.parseDouble(tokens[i + 1]);  
                    vals[i] = vals[i] / 255.0;
                    vals[i] = ((double) (Math.round(vals[i] * 100.0))) / 100.0;
                }
                
//                if (((label == 0) && (count0 <=1755)) || ((label == 1) && (count1 <=1000))) {
                    labels.add(label);
                    images.add(vals);
//                }

                current = b.readLine();
                
            }
            f.close();
            b.close();
            //System.out.println ("Labels "  + count0 + ", " + count1 + ", "  + labels.size() + ", " + images.size());
        }
        catch(Throwable e) {
            e.printStackTrace();
        }
        
    }
    public void processTest(String filename) {
        testImages = new ArrayList<double[]>();
        
        try {
            FileReader f = new FileReader(filename);
            BufferedReader b = new BufferedReader(f);
            String current = b.readLine();
            
            while (current != null) {
                String[] tokens = current.split(",");
                double[] vals = new double[tokens.length];               
                for (int i = 0; i < tokens.length; i++) {
                    vals[i] = Double.parseDouble(tokens[i]);  
                    vals[i] = vals[i] / 255.0;
                    vals[i] = ((double) (Math.round(vals[i] * 100.0))) / 100.0;
                }
                testImages.add(vals);
                current = b.readLine();
                
            }
            f.close();
            b.close();
        }
        catch(Throwable e) {
            e.printStackTrace();
        }
        
    }

    public double computeAI(double yHat) {
        double ai = 0.0;
        ai = 1 / (1 + Math.exp(0.0 - yHat));
        
        if (ai < 0.01) {
            ai = 0.01;
        }
        else if (ai > 0.99) {
            ai = 0.99;
        }
        else {
           //ai = ((double) (Math.round(ai * 10000.0))) / 10000.0;
        }
        return ai;           
    }
    
    public void initWeights() {
        // init
        Random rand = new Random(System.currentTimeMillis());
        for (int i=0; i < W1.length; i++) {
            for (int j=0; j < W1[i].length; j++) {
                W1[i][j] = rand.nextDouble(-1.0, 1.0);
            }
        }
        for (int i=0; i < W2.length; i++) {
            W2[i][0] = rand.nextDouble(-1.0, 1.0);
        }
        for (int i=0; i < B1.length; i++) {
            B1[i][0] = rand.nextDouble(-1.0, 1.0);
        }
        B2 = rand.nextDouble(-1.0, 1.0);
        
    }
    
    public double[] calcAct1 (double[] xi) {
        double[] al1 = new double[HID];
        for (int j=0; j < HID; j++) {
            double sum = 0.0;
            for (int jj=0; jj < M; jj++) {
                sum += xi[jj] * W1[jj][j];
            }
            sum += B1[j][0];
            al1 [j] = computeAI (sum);
        }
        return al1;

    }
    
    public double calcAct2 (double [] act1) {
        double sum = 0.0;
        for (int j=0; j < HID; j++) {
            sum += act1[j] * W2[j][0];
        }
        sum += B2;
        return computeAI (sum);

    }
    
    public double trainSingle(double [] xi, double yi) {
        
        double [] ai1 = calcAct1 (xi);
        double  ai2 = calcAct2 (ai1);
        
        
        // adjust  weights Layer2
        double gradient  = (ai2 - yi) * ai2 * (1.0 - ai2);
        for (int j=0; j < HID; j++) {
            W2[j][0] = W2[j][0] - (ALPHA * gradient * ai1[j]);
        }
        B2 = B2  -  (ALPHA * gradient);
        
        // adjust weights Layer1
        for (int jj=0; jj < M; jj++) {
            for (int j=0; j < HID; j++) {
                W1[jj][j] = W1[jj][j] - (ALPHA * gradient *  W2[j][0] * ai1[j] * (1.0 - ai1[j]) * xi[jj]);
            }
        }
        for (int j=0; j < HID; j++) {
            B1[j][0] = B1[j][0] - (ALPHA * gradient * W2[j][0] * ai1[j] * (1.0 - ai1[j]));
        }
        
        //Layer.printArr(Arrays.toString(W2));
        
        return ai2;
        
    }
    
    public void train() {
        
        int trainSize = images.size();
        int max = 100;
        for (int j = 0; j < max; j++) {
//            Date d = new Date();
//            System.out.println (d + " Starting " + j + " of " + max + " for " + trainSize);
            Collections.shuffle(images);            
            for (int i=0; i < trainSize; i++) {
                double yi = labels.get(i);
                double ai2 = trainSingle (images.get(i), yi);
            }
        }
    }
     
    public void test() {
        
        // AL1
        int testSize = testImages.size();
        double[][] act1 = new double[testSize][HID];
        for (int i=0; i < testSize; i++) {
            double [] xi = testImages.get(i);
            act1[i] = calcAct1(xi);
        }
        
        // AL2
        double[] act2 = new double[testSize];
        for (int i=0; i < act2.length; i++) {
            act2 [i] = calcAct2(act1[i]);
        }
        
        int [] predicted = new int[testSize];
        for (int i=0; i < testSize; i++) {
            if (act2[i] >= 0.5)
                predicted[i] = 1;
        }
        
        // round
        for (int i=0; i < M; i++) {
            for (int j=0; j < HID; j++) {
                W1[i][j] = ((double) (Math.round(W1[i][j] * 10000.0))) / 10000.0; 
            }
        }
        for (int i=0; i < HID; i++) {
            W2[i][0] = ((double) (Math.round(W2[i][0] * 10000.0))) / 10000.0; 
            B1[i][0] = ((double) (Math.round(B1[i][0] * 10000.0))) / 10000.0; 
        }
        B2 = ((double) (Math.round(B2 * 10000.0))) / 10000.0; 
        for (int i=0; i < testImages.size(); i++) {
            act2[i] = ((double) (Math.round(act2[i] * 100.0))) / 100.0; 
        }
        
        // print
        System.out.println ("W1==============");
        for (int i=0; i < W1.length; i++) {
            printArr(Arrays.toString(W1[i]));
        }
        System.out.println ("B1==============");
        Matrix m = new Matrix(B1);
        m = m.transpose();
        double[][] B11 = m.getArray();
        for (int i=0; i < B11.length; i++) {
            printArr(Arrays.toString(B11[i]));
        }
        System.out.println ("W2==============");
        for (int i=0; i < W2.length; i++) {
            printArr(Arrays.toString(W2[i]));
        }
        System.out.println ("B2 ==============");
        System.out.println (B2);
        System.out.println ("==============");
        
        System.out.println ("Test 1==============");
        printArr(Arrays.toString(act1[0]));
        double sum = 0.0;
        for (int i=0; i < act1[0].length;i++) {
            System.out.print((act1[0][i] * W2[i][0]) + ", ");
            sum += (act1[0][i] * W2[i][0]);
        }
        System.out.println("sum = " + sum);
                
        System.out.println("Final act2 = " + act2[0]);
        System.out.println ("Test 1==============");


        printArr(Arrays.toString(act2));
        printArr(Arrays.toString(predicted));

    }


}
