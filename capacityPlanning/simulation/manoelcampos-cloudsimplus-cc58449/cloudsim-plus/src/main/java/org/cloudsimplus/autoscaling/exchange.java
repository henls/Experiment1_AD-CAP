package org.cloudsimplus.autoscaling;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.RandomAccessFile;
import java.nio.channels.FileChannel;
import java.nio.channels.FileLock;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.collections4.ListUtils;
import org.cloudbus.cloudsim.cloudlets.CloudletExecution;
import org.cloudbus.cloudsim.vms.Vm;

import org.cloudsimplus.autoscaling.LimitQueue;
import org.cloudsimplus.autoscaling.SlaStatistic;


import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONObject;
import org.apache.commons.io.FileUtils;

public class exchange {
    String configPath = "/home/wangxinhua/Experiment1_AD-CAP/RL-Transformer/config/configure";
    String action = new String();
    String status = new String();
    int vmSize = 0;
    int limit = 10;
    JSONObject json = null;
    LimitQueue<Double> queue = new LimitQueue<Double>(limit);
    List<CloudletExecution> completeLast = new ArrayList<CloudletExecution>();
    double RT = 0;
    public exchange(){
    File configFile = new File(configPath);
        if (configFile.exists() == false){
            System.out.println("#INFO: configure not found");
            System.exit(0);
        }
        try {
            String content = FileUtils.readFileToString(configFile, "UTF-8");
            json = JSON.parseObject(content);
            this.action = json.getString("action");
            this.status = json.getString("status");
            this.vmSize = json.getIntValue("vmSize");
        } catch (Exception e) {
            System.out.println(e);
        }
    }

    public String readAction(){
        String content = "OK";
        try {
            while (content.contains("OK") == true){
                content = lockRead(this.action);
            }
            lockWrite(this.action, content + "OK");
            return content;
        } catch (Exception e) {
            System.out.println(e);
            System.out.println("INFO# something error when reading action.");
            return "";
        }
    }
    public void writeStatus(Vm vm){
        //所有虚拟机增减完成后才发送reward，因为要做水平伸缩的实验。暂时只考虑静态虚拟机个数。
        List<Vm> vmList = new ArrayList<>();
        vmList.add(vm);
        SlaStatistic sla = new SlaStatistic(vmList);
        String content = new String();
        String statusSpace = getStatus(vm);
        String reward = new String();
        if (statusSpace == "null"){
            reward = "null";
        }else{
            reward = sla.getViloate(vm, true);
        }
        String done = new String();
        if (vm.getCloudletScheduler().getCloudletExecList().size() == 0){
            done = "1";
        }else{
            done = "0";
        }
        try {
            while (content.contains("OK") == false){
                content = lockRead(this.status);
            }
            lockWrite(this.status, statusSpace + "&" + reward + "&" + done);
        } catch (Exception e) {
            System.out.println(e);
            System.out.println("INFO# something error when write file.");
        }
    }
    public String getStatus(Vm vm){
        double cpuPercentage = vm.getCpuPercentUtilization();
        if (Double.isNaN(cpuPercentage) != true){
            if (queue.size() > 0){
                if (queue.getLast() != cpuPercentage){
                    queue.offer(cpuPercentage);
                    }
            }else{
                queue.offer(cpuPercentage);
            }
        }
        List<CloudletExecution> completeList = vm.getCloudletScheduler().getCloudletFinishedList();
        List<CloudletExecution> newComplete = ListUtils.subtract(completeList,completeLast);
        List<CloudletExecution> completeLast = new ArrayList<CloudletExecution>(completeList);
        this.completeLast = completeLast;
        double tmp = 0.;
        for (CloudletExecution newCloudlet : newComplete) {
            double execTime = newCloudlet.getFinishTime() - newCloudlet.getCloudletArrivalTime();
            tmp = tmp + execTime;
        }
        if (newComplete.size() != 0){
            this.RT = tmp / newComplete.size();
        }
        
        long totalPEs = vm.getNumberOfPes();
        long availablePEs = vm.getFreePesNumber();
        long usedPEs = totalPEs - availablePEs;
        if (queue.size() == limit){
            return queue.get() + "$" + RT+ "$" + usedPEs + "$" + totalPEs;
        }else{
            return "null";
        }
    }
    public String lockRead(String pth){
        File file=new File(pth);  
        try {
            if(!file.exists()){
                file.createNewFile();
            }

            RandomAccessFile randomAccessFile = new RandomAccessFile(file, "rw");
            FileChannel fileChannel=randomAccessFile.getChannel();
            FileLock fileLock=null;
            while(true){
                try {
                    fileLock = fileChannel.tryLock();
                    break;
                } catch (Exception e) {
                    e.printStackTrace();
                    System.out.println("lock is own by python");
                    Thread.sleep(1);
                }
            }
            
            byte[] buf = new byte[1024];  
            randomAccessFile.read(buf);
            fileLock.release();
            fileChannel.close();
            randomAccessFile.close();
            randomAccessFile=null;
            return new String(buf, "utf-8");
        } catch (IOException e) {
            e.printStackTrace();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        return "OK";
    }

    public void lockWrite(String pth, String content){
        File file=new File(pth);  
        try {
            if(!file.exists()){
                file.createNewFile();
            }
            RandomAccessFile randomAccessFile = new RandomAccessFile(file, "rw");
            randomAccessFile.setLength(0);
            FileChannel fileChannel=randomAccessFile.getChannel();
            FileLock fileLock=null;
            while(true){
                try {
                    fileLock = fileChannel.tryLock();
                    break;
                } catch (Exception e) {
                    e.printStackTrace();
                    System.out.println("lock is own by python");
                    Thread.sleep(1);
                }
            }
            //randomAccessFile.write(content.getBytes("utf-8"));
            content = trimnull(content);
            randomAccessFile.writeBytes(content);
            fileLock.release();
            fileChannel.close();
            randomAccessFile.close();
            randomAccessFile=null;
        } catch (IOException e) {
            e.printStackTrace();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    /**
	 * 去除字符串中的null域
	 * @param string
	 * @return
	 * @throws UnsupportedEncodingException 
	 */
	public static String trimnull(String string)
    {   try {
        ArrayList<Byte> list = new ArrayList<Byte>();
		byte[] bytes = string.getBytes("UTF-8");
		for(int i=0;bytes!=null&&i<bytes.length;i++){
			if(0!=bytes[i]){
				list.add(bytes[i]);
			}
		}
		byte[] newbytes = new byte[list.size()];
		for(int i = 0 ; i<list.size();i++){
			newbytes[i]=(Byte) list.get(i); 
		}
		String str = new String(newbytes,"UTF-8");
	    return str;
    } catch (Exception e) {
        return "";
    }
		
    }
}   