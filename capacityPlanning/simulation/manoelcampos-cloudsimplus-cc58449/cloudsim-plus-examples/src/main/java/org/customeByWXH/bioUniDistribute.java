package org.customeByWXH;
import java.util.ArrayList;
import java.util.List;

public class bioUniDistribute{

    public static List<Object[]> calculate(ArrayList<Double> items, int index, List<Object[]> resultList){ 
        List<Object[]> dataList=new ArrayList<Object[]>();
        int k = items.size();
        for (int i = 0; i < k; i++){
            dataList.add(new Object[]{ String.valueOf(items.get(i)), String.valueOf(1 - items.get(i))});
        }
        if(index==dataList.size()){
            return resultList;
        }
        
        List<Object[]> resultList0=new ArrayList<Object[]>();
        if(index==0){
            Object[] objArr=dataList.get(0);
            for(Object obj : objArr){
                resultList0.add(new Object[]{obj});
            }
        }else{
            Object[] objArr=dataList.get(index);
            for(Object[] objArr0: resultList){
                for(Object obj : objArr){
                    //复制数组并扩充新元素
                    Object[] objArrCopy=new Object[objArr0.length+1];
                    System.arraycopy(objArr0, 0, objArrCopy, 0, objArr0.length);
                    objArrCopy[objArrCopy.length-1]=obj;
                    
                    //追加到结果集
                    resultList0.add(objArrCopy);
                }
            }
        }
        return calculate(items, ++index, resultList0);
    }
    
    /*public static void main(String[] args) {
        ArrayList<Double> usageList = new ArrayList<Double>();
        usageList.add(0.5);
        usageList.add(0.1);
        List<Object[]> resultList= calculate(usageList ,0,null); 
        //k是此cpu上同时运行的任务数
        //打印组合结果 
        for(int i=0;i<resultList.size();i++){
            Object[] objArr=resultList.get(i);
            System.out.print("\n组合"+(i+1)+"---");
            for(Object obj : objArr){
                System.out.print( obj+" "); 
            }
        } 
    }*/
}
