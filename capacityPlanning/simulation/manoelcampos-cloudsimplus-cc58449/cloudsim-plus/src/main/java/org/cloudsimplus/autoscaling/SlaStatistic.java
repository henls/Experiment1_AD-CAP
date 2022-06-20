package org.cloudsimplus.autoscaling;

import org.apache.commons.collections4.ListUtils;
import org.cloudbus.cloudsim.cloudlets.CloudletExecution;
import org.cloudbus.cloudsim.vms.Vm;
import org.cloudsimplus.listeners.EventInfo;

import java.util.List;
import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Arrays;

public class SlaStatistic {
    /*
        默认参数列表
    */
    int RT_MAX = 300; //RT:response time and unit is senond
    int RT_MIN = 200;
    int UTILIZATION_MAX = 80;
    int UTILIZATION_MIN = 40;
    double TOTAL_SAMPLE = 0. + 1e-8;//总的采样数
    double VIOLATE_SAMPLE = 0.;//统计的时段内发生违反的次数
    int SAMPLE_RATE = 1;//采样率1s一次
    int lastTime;
    private static final String workdir = "/home/wangxinhua/Experiment1_AD-CAP" + "/capacityPlanning/simulation/manoelcampos-cloudsimplus-cc58449/cloudsim-plus-examples/src/main/";

    String pth = workdir + "output/sla.csv";

    List<CloudletExecution> completeLast = new ArrayList<CloudletExecution>();
    List<Vm> vmList = new ArrayList<Vm>();

    public SlaStatistic(List<Vm> vmList) {
        this.vmList = vmList;
    }

    public void getViloate(EventInfo evt){

        int nowTime = (int) evt.getTime();
        if (nowTime % SAMPLE_RATE == 0 && nowTime != lastTime){
            lastTime = nowTime;
            for (Vm vm : vmList) {
                List<CloudletExecution> completeList = vm.getCloudletScheduler().getCloudletFinishedList();
                List<CloudletExecution> newComplete = ListUtils.subtract(completeList,completeLast);
                List<CloudletExecution> completeLast = new ArrayList<CloudletExecution>(completeList);
                this.completeLast = completeLast;
                int vmUtil = (int) vm.getCpuPercentUtilization() * 100;
                for (CloudletExecution newCloudlet : newComplete) {
                    TOTAL_SAMPLE += 1.;
                    double execTime = newCloudlet.getFinishTime() - newCloudlet.getCloudletArrivalTime();
                    if (execTime < RT_MIN || execTime > RT_MAX || vmUtil < UTILIZATION_MIN || vmUtil > UTILIZATION_MAX){
                        /*System.out.println("execTime " + execTime);
                        System.out.printf("execTime < RT_MIN %b || execTime > RT_MAX %b || vmUtil < UTILIZATION_MIN %b || vmUtil > UTILIZATION_MAX %b%n",
                                            execTime < RT_MIN, 
                                            execTime > RT_MAX, 
                                            vmUtil < UTILIZATION_MIN, 
                                            vmUtil > UTILIZATION_MAX
                                            );*/
                        VIOLATE_SAMPLE += 1.;
                    }
                }
            }
            //System.out.printf("#INFO violate rate is %2.2f%% %n", VIOLATE_SAMPLE / TOTAL_SAMPLE * 100);
            writeSla(pth, "" + VIOLATE_SAMPLE / TOTAL_SAMPLE + "\n");
        }
    }
    //单个虚拟机上的违反率计算，vmList传入的长度为1
    public String getViloate(Vm vms, boolean reward){
        int nowTime = (int) vms.getSimulation().clock();
        if (nowTime % SAMPLE_RATE == 0 && nowTime != lastTime){
            lastTime = nowTime;
            for (Vm vm : vmList) {
                List<CloudletExecution> completeList = vm.getCloudletScheduler().getCloudletFinishedList();
                List<CloudletExecution> newComplete = ListUtils.subtract(completeList,completeLast);
                List<CloudletExecution> completeLast = new ArrayList<CloudletExecution>(completeList);
                this.completeLast = completeLast;
                int vmUtil = (int) vm.getCpuPercentUtilization() * 100;
                for (CloudletExecution newCloudlet : newComplete) {
                    TOTAL_SAMPLE += 1.;
                    double execTime = newCloudlet.getFinishTime() - newCloudlet.getCloudletArrivalTime();
                    if (execTime < RT_MIN || execTime > RT_MAX || vmUtil < UTILIZATION_MIN || vmUtil > UTILIZATION_MAX){
                        /*System.out.println("execTime " + execTime);
                        System.out.printf("execTime < RT_MIN %b || execTime > RT_MAX %b || vmUtil < UTILIZATION_MIN %b || vmUtil > UTILIZATION_MAX %b%n",
                                            execTime < RT_MIN, 
                                            execTime > RT_MAX, 
                                            vmUtil < UTILIZATION_MIN, 
                                            vmUtil > UTILIZATION_MAX
                                            );*/
                        VIOLATE_SAMPLE += 1.;
                    }
                }
            }
            //System.out.printf("#INFO violate rate is %2.2f%% %n", VIOLATE_SAMPLE / TOTAL_SAMPLE * 100);
            return VIOLATE_SAMPLE / TOTAL_SAMPLE + "";
        }
        return "";
    }

    public void writeSla(String pth, String cont){
        try {
            File file = new File(pth);
            OutputStream output = new FileOutputStream(file, true);
            output.write(cont.getBytes());
            output.close();
        } catch (Exception e) {
            System.out.println("Exception thrown  :" + e);
        }
    }
}
