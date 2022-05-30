package org.cloudsimplus.examples.autoscaling;

import org.cloudbus.cloudsim.cloudlets.Cloudlet;
import org.cloudbus.cloudsim.cloudlets.CloudletSimple;
import org.cloudbus.cloudsim.utilizationmodels.UtilizationModel;
import org.cloudbus.cloudsim.utilizationmodels.UtilizationModelDynamic;
import org.cloudbus.cloudsim.utilizationmodels.UtilizationModelFull;
import org.cloudbus.cloudsim.utilizationmodels.UtilizationModelPlanetLab;

import java.util.Random;
import java.util.HashMap;

public class  AnomalyModel {
    String[] traceCpu;
    String[] traceRam;
    Random getRand;
    int VM_PES;
    int scheduleInterval;
    public AnomalyModel(String[] traceCpuAnomaly, String[] traceRamAnomaly, Random getRand, int scheduleInterval, int VM_PES) {
        this.traceCpu = traceCpuAnomaly;
        this.traceRam = traceRamAnomaly;
        this.getRand = getRand;
        this.scheduleInterval = scheduleInterval;
        this.VM_PES = VM_PES;
    }
    
    public Cloudlet generateAnomaly(long length, int numberOfPes, final String resource, final String type, final double delay) {
        /*
        Since a VM PE isn't used by two Cloudlets at the same time,
        the Cloudlet can used 100% of that CPU capacity at the time
        it is running. Even if a CloudletSchedulerTimeShared is used
        to share the same VM PE among multiple Cloudlets,
        just one Cloudlet uses the PE at a time.
        Then it is preempted to enable other Cloudlets to use such a VM PE.
         */
        final UtilizationModel utilizationCpu = getUtilizationModel(resource, type).get("cpu");
        
        final UtilizationModel utilizationRam = getUtilizationModel(resource, type).get("ram");
        
        /**
         * Since BW e RAM are shared resources that don't enable preemption,
         * two Cloudlets can't use the same portion of such resources at the same time
         * (unless virtual memory is enabled, but such a feature is not available in simulation).
         * This way, the total capacity of such resources is being evenly split among created Cloudlets.
         * If there are 10 Cloudlets, each one will use just 10% of such resources.
         * This value can be defined in different ways, as you want. For instance, some Cloudlets
         * can require more resources than other ones.
         * To enable that, you would need to instantiate specific {@link UtilizationModelDynamic} for each Cloudlet,
         * use a {@link UtilizationModelStochastic} to define resource usage randomly,
         * or use any other {@link UtilizationModel} implementation.
        */
        if (type.equals("burst")){
            
            length = new Integer(1000);
            numberOfPes = new Integer(VM_PES);
        }
        if (type.equals("bottleneck")){
            numberOfPes = new Integer(VM_PES);
            
        }
        final UtilizationModel utilizationModelDynamic = new UtilizationModelDynamic(1.0/1000);
        Cloudlet cl = new CloudletSimple(length, numberOfPes);
        cl.setFileSize(1024)
            .setOutputSize(1024)
            .setUtilizationModelBw(utilizationModelDynamic)
            .setUtilizationModelRam(utilizationRam)
            .setUtilizationModelCpu(utilizationCpu)
            .setSubmissionDelay(delay);
        return cl;
    }

    private HashMap<String, UtilizationModel> getUtilizationModel(String resource, String type){

        UtilizationModel modelCPU = new UtilizationModelFull();
        UtilizationModel modelRAM = new UtilizationModelFull();

        HashMap<String, UtilizationModel> modelsDict = new HashMap<String, UtilizationModel>();
        //System.out.println(type);
        switch(type){
            case "incremental"://利用率逐渐增加
                modelCPU = new UtilizationModelDynamic(1.)
                            .setMaxResourceUtilization(100)
                            .setUtilizationUpdateFunction(this::utilizationIncrement);
                modelRAM = new UtilizationModelDynamic(0.)
                            .setMaxResourceUtilization(0.1)
                            .setUtilizationUpdateFunction(this::utilizationIncrementRam);
                break;
            case "patternShift"://放入新的trace
                int number = getRand.nextInt(4); 
                modelCPU = UtilizationModelPlanetLab.getInstance(traceCpu[number], scheduleInterval);
                modelRAM = UtilizationModelPlanetLab.getInstance(traceRam[number], scheduleInterval);
                
                break;
            case "burst"://放入短时间资源需求高的负载
            modelCPU = new UtilizationModelDynamic(1.)
                        .setMaxResourceUtilization(100)
                        .setUtilizationUpdateFunction(this::utilizationIncrementBurst);
            modelRAM = new UtilizationModelDynamic(0.)
                        .setMaxResourceUtilization(0.1)
                        .setUtilizationUpdateFunction(this::utilizationIncrementRamBurst);
                break;
            case "bottleneck"://增加瓶颈异常，任务大小正常
            modelCPU = new UtilizationModelDynamic(0.)
                        .setMaxResourceUtilization(100)
                        .setUtilizationUpdateFunction(this::utilizationIncrementBottleNeck);
            modelRAM = new UtilizationModelDynamic(0.)
                        .setMaxResourceUtilization(0.2)
                        .setUtilizationUpdateFunction(this::utilizationIncrementRamBurst);
                break;
            
        }
        UtilizationModel nulCpu = new UtilizationModelDynamic(0.1)
                                .setMaxResourceUtilization(100);
        UtilizationModel nulRam = new UtilizationModelDynamic(0.01)
                                .setMaxResourceUtilization(1);
        switch(resource){
            case "cpu":
                modelsDict.put("cpu", modelCPU);
                modelsDict.put("ram", nulRam);
                break;
            case "ram":
                modelsDict.put("ram", modelRAM);
                modelsDict.put("cpu", nulCpu);
                break;
            case "mix":
                modelsDict.put("cpu", modelCPU);
                modelsDict.put("ram", modelRAM);
                break;
        }
        return modelsDict;
    }

    private double utilizationIncrement(UtilizationModelDynamic um) {
        return um.getUtilization() + um.getTimeSpan()*10;
    }

    private double utilizationIncrementRam(UtilizationModelDynamic um) {
        return um.getUtilization() + 0.001;
    }

    private double utilizationIncrementBurst(UtilizationModelDynamic um) {
        
        return um.getUtilization() + um.getTimeSpan()*50;
    }

    private double utilizationIncrementBottleNeck(UtilizationModelDynamic um) {
        
        return um.getUtilization() + um.getTimeSpan()*90;
    }

    private double utilizationIncrementRamBurst(UtilizationModelDynamic um) {
        return um.getUtilization() + 0.1;
    }

}
