package org.customeByWXH;
import java.util.ArrayList;
import java.util.List;

public class combinations {

    public static List<Object[]> combination(int k, int index, List<Object[]> resultList){ 
        List<Object[]> dataList=new ArrayList<Object[]>();
        for (int i = 0; i < k; i++){
            dataList.add(new Object[]{"1", "0"});
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
        return combination(k,++index,resultList0);
    }
    
    /*public static void main(String[] args) {
        int k = 2;
        List<Object[]> resultList= combination(k ,0,null); 
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
