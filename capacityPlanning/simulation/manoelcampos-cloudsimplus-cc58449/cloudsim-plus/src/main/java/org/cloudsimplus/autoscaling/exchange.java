package org.cloudsimplus.autoscaling;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
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
        File actID = new File(this.action);
        if (actID.exists() == false){
            try {
                actID.createNewFile();
            } catch (Exception e) {
                System.out.println(e);
            }
        }
        String content = "OK";
        try {
            while (content.contains("OK") == true){
                InputStream input = new FileInputStream(actID);
                byte config[] = new byte[1024];
                int len = input.read(config);
                input.close();
                content = new String(config, 0, len);
            }
            OutputStream output = new FileOutputStream(actID);
            output.write((content + "OK").getBytes());
            output.close();
            return content;
        } catch (Exception e) {
            System.out.println(e);
            System.out.println("INFO# something error when reading action.");
            return "";
        }
    }
    public void writeStatus(Vm vm){
        //所有虚拟机增减完成后才发送reward，因为要做水平伸缩的实验。暂时只考虑静态虚拟机个数。
        File statusID = new File(this.status);
        List<Vm> vmList = new ArrayList<>();
        vmList.add(vm);
        SlaStatistic sla = new SlaStatistic(vmList);
        if (statusID.exists() == false){
            try {
                statusID.createNewFile();
            } catch (Exception e) {
                System.out.println(e);
            }
        }
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
                InputStream input = new FileInputStream(statusID);
                byte config[] = new byte[1024];
                int len = input.read(config);
                input.close();
                content = new String(config, 0, len);
            }
            OutputStream output = new FileOutputStream(statusID);
            output.write((statusSpace + "&" + reward + "&" + done).getBytes());
            output.close();
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
        double min = 1.e10;
        double max = 0.;
        for (CloudletExecution newCloudlet : newComplete) {
            double execTime = newCloudlet.getFinishTime() - newCloudlet.getCloudletArrivalTime();
            if (execTime > max){
                max = execTime;
            }
            if (execTime < min){
                min = execTime;
            }
        }
        long totalPEs = vm.getNumberOfPes();
        long availablePEs = vm.getFreePesNumber();
        long usedPEs = totalPEs - availablePEs;
        if (queue.size() == limit){
            return queue.get() + "$" + max + "$" + min + "$" + usedPEs + "$" + totalPEs;
        }else{
            return "null";
        }
    }
}   
