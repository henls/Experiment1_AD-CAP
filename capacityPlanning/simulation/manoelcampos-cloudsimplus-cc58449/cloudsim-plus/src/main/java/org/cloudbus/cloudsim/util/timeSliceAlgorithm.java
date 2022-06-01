package org.cloudbus.cloudsim.util;
import java.util.ArrayList;
import java.util.List;
//import org.customeByWXH.bioUniDistribute;
//import org.customeByWXH.combinations;
//import org.cloudbus.cloudsim.util.combinations;
//import org.cloudbus.cloudsim.util.bioUniDistribute;

public class timeSliceAlgorithm{

    public static void main(String[] args) {
        ArrayList<Double> usage = new ArrayList<Double>();
        //usage.add(0.);
        //usage.add(0.1);
        //usage.add(0.5);
        double ans = 0;
        ans = utilization(usage);
        System.out.println(ans);
    }

    public static double utilization(ArrayList<Double> usages){
        int k = usages.size();
        if (k == 0){
            usages.add(0.);
            k = 1;
        }
        List<Object[]> Matrix = combinations.combination(k, 0, null);
        List<Object[]> UniPdf = bioUniDistribute.calculate(usages, 0, null);
        double used = 0.;
        double total = 0.;
        double zeroProb = 1.;;
        for(int i=0;i<Matrix.size();i++){
            Object[] objMatrix = Matrix.get(i);
            Object[] objUniPdf = UniPdf.get(i);
            int tmpMat = 0;
            for(Object obj : objMatrix){
                tmpMat = (int) tmpMat + Integer.parseInt(obj.toString());
            }
            double tmpPdf = 1.0;
            for(Object obj : objUniPdf){
                tmpPdf = tmpPdf * Double.parseDouble(obj.toString());
                zeroProb = tmpPdf;
            }
            used = used + tmpMat * tmpPdf;
        } 
        total = used + zeroProb;

        return used / total;
    };
        
}