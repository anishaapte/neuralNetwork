import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import Jama.Matrix;

public class Compute {
    public static List<Integer> labels;
    public static int[] testLabels;
    public List<double[]> images;
    public List<double[]> testImages;
    public static final double ALPHA = 0.001;
    public static final int NUM_COLS = 784;
    public static final int HID_UNITS = 28;
    public double[] weights;
    public double bias;
    public static final double DIF = 0.999;
    public static final double MAX_COST = 10000.0;
    public static final int IMAGE_SIZE = 2000;
    public static final double COST_THRESH = 0.001;
    public static double[] testAct;
    
    public static void main(String[] args) {
        Compute c = new Compute();
        c.processText("/Users/anishaapte/Desktop/AI540/mnist_test.csv");
        c.processTest("/Users/anishaapte/Desktop/AI540/regTest.txt");
        c.initWeightAndBias();
        c.gradientDescent();
        c.getTestAct();
        System.out.println(Arrays.toString(testAct));
        System.out.println(Arrays.toString(testLabels));
  }
    public static void printArr(String s) {
        System.out.println(s.substring(1, s.length() - 1));
    }
    
    public Compute() {
        
    }
    
    public void processText(String filename) {
        labels = new ArrayList<Integer>();
        images = new ArrayList<double[]>();
        
        try {
            FileReader f = new FileReader(filename);
            BufferedReader b = new BufferedReader(f);
            String current = b.readLine();
            
            while (current != null) {
                String[] tokens = current.split(",");
                
                //setting the label
                int label = Integer.parseInt(tokens[0]);               
                if (label == 2) {
                    label = 1;
                }
                else {
                    label = 0;
                }
                labels.add(label);
                
                //adding remaining columns 
                double[] vals = new double[tokens.length - 1];               
                for (int i = 0; i < tokens.length - 1; i++) {
                    vals[i] = Double.parseDouble(tokens[i + 1]);  
                    vals[i] = vals[i] / 255.0;
                    vals[i] = ((double) (Math.round(vals[i] * 100.0))) / 100.0;
                }
                images.add(vals);
                current = b.readLine();
                
//                for (double[] row: images) {
//                    System.out.println(Arrays.toString(row));
//                }
            }
            f.close();
            b.close();
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
    public void initWeightAndBias() {
        weights = new double[NUM_COLS];
        Random r = new Random();
        for (int i = 0; i < weights.length; i++) {
           weights[i] = r.nextDouble(-1.0, 1.0);
        }
        bias = r.nextDouble(-1.0, 1.0);
        
    }
    
    public double processOneIter() {
        double[] summation = new double[NUM_COLS];
        double sumB = 0.0;
        double totalCost = 0.0;
        
        //calculating activation and delta
        for (int i = 0; i < IMAGE_SIZE; i++) {
            double[] rowImage = images.get(i);
            double yhat = computeYhat(weights, rowImage,bias);
            double ai = computeAI(yhat);
            double yi = ((double) labels.get(i));
            totalCost += computeCost(ai, yi);
            
            for (int j = 0; j < NUM_COLS; j++) {
                summation[j] += (ai - yi) * rowImage[j];
            }
            sumB+= (ai - yi);
        }
        
        //updating weights and bias
        for (int j = 0; j < NUM_COLS; j++) {
            weights[j] = weights[j] - (ALPHA * summation[j]);
        }
        bias = bias - (ALPHA * sumB);
        return totalCost;
  
    }
    
    public void gradientDescent() {  
        double preCost = 0.0;
        double cost = processOneIter();
        int count = 0;
        while (Math.abs(preCost - cost) > COST_THRESH) {
            preCost = cost;
            cost = processOneIter();
            if (true) {
                //System.out.println(count + " ----- " + cost);
            }
            count++;
                
        }
        
        //System.out.println("--------");
        for (int i = 0; i < weights.length; i++) {
            weights[i] = ((double) (Math.round(weights[i] * 10000.0))) / 10000.0;           
        }
        System.out.println(Arrays.toString(weights));
        bias = ((double) (Math.round(bias * 10000.0))) / 10000.0;  
        System.out.println(bias);
    }
    
    public double computeYhat (double[] weights, double[] image, double bias) {
        double yHat = 0.0;
        
        for (int i = 0; i < weights.length; i++) {
            yHat+= (weights[i] * image[i]);
        }
        yHat += bias;
        return yHat;
        
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
    
    public double computeCost(double ai, double yi) {
        double cost = 0.0;
        double diff = Math.abs(ai-yi);
        if (diff >= DIF) {
            System.out.println ("MAX .. ");
            cost = MAX_COST;
        } else {
            if (yi == 0) {
                cost = -1.0 * (Math.log(1.0-ai));
            }
            else {
                cost = -1.0 * (Math.log(ai));
            }
        }
        //System.out.println("ai, yi, cost " + ai + " " + yi + " " + cost);
        return cost;               
    }
    
    public void getTestAct() {
        testAct = new double[testImages.size()];
        testLabels = new int[testImages.size()];
        for(int i = 0; i < testAct.length; i++) {
            double yhat = computeYhat(weights, testImages.get(i), bias);
            double ai = computeAI(yhat);
            testAct[i] = ((double) (Math.round(ai * 10000.0))) / 10000.0;  
            if (testAct[i] >= 0.5) {
                testLabels[i] = 1;
            }
        }
        
        
    }
    

 
    
  
}
