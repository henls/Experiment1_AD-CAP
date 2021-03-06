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
    int RT_MAX = 100; //RT:response time and unit is senond
    int RT_MIN = 30;
    int UTILIZATION_MAX = 80;
    int UTILIZATION_MIN = 10;
    double TOTAL_SAMPLE = 0. + 1e-8;//总的采样数
    double VIOLATE_SAMPLE = 0.;//统计的时段内发生违反的次数
    int SAMPLE_RATE = 10;//采样率1s一次
    int lastTime;
    int rt_record = 0;
    private static final String workdir = "/home/wangxinhua/Experiment1_AD-CAP" + "/capacityPlanning/simulation/manoelcampos-cloudsimplus-cc58449/cloudsim-plus-examples/src/main/";

    String pth = workdir + "output/sla.csv";

    List<CloudletExecution> completeLast = new ArrayList<CloudletExecution>();
    List<Vm> vmList = new ArrayList<Vm>();
    int lastCloudletsNumber = 0;
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
                    if (execTime < RT_MIN || execTime > RT_MAX || vmUtil > UTILIZATION_MAX){
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
            System.out.printf("#INFO violate rate is %2.2f%% %n", VIOLATE_SAMPLE / TOTAL_SAMPLE * 100);
            writeSla(pth, "" + VIOLATE_SAMPLE / TOTAL_SAMPLE + "\n");
        }
    }
    //单个虚拟机上的违反率计算，vmList传入的长度为1 用于与RL模型联动。
    public String getViloate(Vm vms, boolean reward){
        int nowTime = (int) vms.getSimulation().clock();
        if (nowTime % SAMPLE_RATE == 0 && nowTime != lastTime){
            double rt_reward = 0.;
            double ut_reward = 1.;
            lastTime = nowTime;
            int tmp = 0;
            List<Vm> vmListFake = new ArrayList<Vm>();
            vmListFake.add(vms);
            double vmUtil = 0.;
            for (Vm vm : vmListFake) {
                List<CloudletExecution> completeList = vm.getCloudletScheduler().getCloudletFinishedList();
                List<CloudletExecution> newComplete = ListUtils.subtract(completeList,completeLast);
                List<CloudletExecution> completeLast = new ArrayList<CloudletExecution>(completeList);
                this.completeLast = completeLast;
                //vmUtil = vm.getCpuPercentUtilization() * 100;
                vmUtil = getVmCpuMean(vm, nowTime);
                for (CloudletExecution newCloudlet : newComplete) {
                    TOTAL_SAMPLE += 1.;
                    double execTime = newCloudlet.getFinishTime() - newCloudlet.getCloudletArrivalTime();
                    if (execTime < RT_MIN || execTime > RT_MAX || vmUtil > UTILIZATION_MAX){
                        /*System.out.println("execTime " + execTime);
                        System.out.printf("execTime < RT_MIN %b || execTime > RT_MAX %b || vmUtil < UTILIZATION_MIN %b || vmUtil > UTILIZATION_MAX %b%n",
                                            execTime < RT_MIN, 
                                            execTime > RT_MAX, 
                                            vmUtil < UTILIZATION_MIN, 
                                            vmUtil > UTILIZATION_MAX
                                            );*/
                        VIOLATE_SAMPLE += 1.;
                    }
                    tmp += execTime;
                }
                //计算当前间隔的平均响应时间
                int rt = 0;
                if (newComplete.size() != 0){
                    rt = tmp / newComplete.size();
                }
                rt_record = rt;
                //响应时间的奖励函数
                rt_reward = 1.;
                if (rt > RT_MAX){
                    rt_reward = Math.pow(Math.E, -1 * (Math.pow((rt - RT_MAX)/RT_MAX, 2)));
                }
                if (rt < RT_MIN){
                    rt_reward = Math.pow(Math.E, -1 * (Math.pow((RT_MIN - rt)/RT_MIN, 2)));
                }
                //虚拟机利用率的奖励函数，正常多个虚拟机多个资源都要做。现在只考虑在单个虚拟机的CPU上做垂直扩展的。
                ut_reward = Math.abs(vmUtil / 100. - UTILIZATION_MAX / 100.) + 1.;
            }
            //System.out.printf("#INFO violate rate is %2.2f%% %n", VIOLATE_SAMPLE / TOTAL_SAMPLE * 100);
            //return VIOLATE_SAMPLE / TOTAL_SAMPLE + "";//保留，在呈现实验结果时sla违反率更直观
            //System.out.printf("|rt: %d|rt_reward: %f|ut_reward: %f|reward: %f|\n",rt_record, rt_reward, ut_reward, rt_reward / ut_reward);
            return rt_reward / ut_reward + "";
            //return "-1";//最短时间完成，验证模型有效性
            //return -1 * vmUtil / 100. + "";
            //return -1 * Math.abs(vmList.get(0).getCloudletScheduler().getCloudletExecList().size() - vmList.get(0).getNumberOfPes()) + "";
            //String ret = -1 * Math.abs(lastCloudletsNumber - vms.getNumberOfPes()) + "";
            //lastCloudletsNumber = vms.getCloudletScheduler().getCloudletExecList().size();
            //return ret;
        }
        return "";
    }

    public double getVmCpuMean(Vm vm, int time){
        int START_TIME = time - SAMPLE_RATE + 1;
        double ans = 0.;
        for (int i = START_TIME; i <= time; i++){
            ans += vm.getCpuPercentUtilization((double) i);
        }
        return ans / SAMPLE_RATE;
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
