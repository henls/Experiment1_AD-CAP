package edu.rice.rubis.client;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Calendar;
import java.text.SimpleDateFormat;
import java.io.RandomAccessFile;
import java.util.*;
/**
 * This class provides thread-safe statistics. Each statistic entry is composed as follow:
 * <pre>
 * count     : statistic counter
 * error     : statistic error counter
 * minTime   : minimum time for this entry (automatically computed)
 * maxTime   : maximum time for this entry (automatically computed)
 * totalTime : total time for this entry
 * </pre>
 *
 * @author <a href="mailto:cecchet@rice.edu">Emmanuel Cecchet</a> and <a href="mailto:julie.marguerite@inrialpes.fr">Julie Marguerite</a>
 * @version 1.0
 */

public class Stats
{
  private int nbOfStats;
  private int count[];
  private int error[];
  private long minTime[];
  private long maxTime[];
  private long totalTime[];
  private int  nbSessions;   // Number of sessions succesfully ended
  private long sessionsTime; // Sessions total duration
  private String time_now[]; // add by wxh
  private String savePath;

  /**
   * Creates a new <code>Stats</code> instance.
   * The entries are reset to 0.
   *
   * @param NbOfStats number of entries to create
   */
  public Stats(int NbOfStats)
  {
    nbOfStats = NbOfStats;
    count = new int[nbOfStats];
    error = new int[nbOfStats];
    minTime = new long[nbOfStats];
    maxTime = new long[nbOfStats];
    totalTime = new long[nbOfStats];
    time_now = new String[nbOfStats];

    
    reset();
    //Create new fold when launch this emulation.  create()
    createFold();
  }

  public void createFold(){
    int i = 0;
    while(true){
      try {
        i++;
        savePath = "/home/wxh/data/" + i; 
        String workPath = "/home/wxh/data";
        File file = new File(savePath);
        long timeDelta = sortFileByName(workPath, "asc");
        if (file.exists() == false && timeDelta > 1 ){
          file.mkdirs();
          break;
        }else if (file.exists() == false && timeDelta < 1){
          break;
        }
        if (i>1000){
          break;
        }
      } catch (Exception e) {

        e.printStackTrace();
      }
      }
    
  }
  /**
   * Resets all entries to 0
   */
  public synchronized void reset()
  {
    int i;

    for (i = 0 ; i < nbOfStats ; i++)
    {
      count[i] = 0;
      error[i] = 0;
      minTime[i] = Long.MAX_VALUE;
      maxTime[i] = 0;
      totalTime[i] = 0;
    }
    nbSessions = 0;
    sessionsTime = 0;
  }

  /**
   * Add a session duration to the total sessions duration and
   * increase the number of succesfully ended sessions.
   *
   * @param time duration of the session
   */
  public synchronized void addSessionTime(long time)
  {
    nbSessions++;
    if (time < 0)
    {
      System.err.println("Negative time received in Stats.addSessionTime("+time+")<br>\n");
      return ;
    }
    sessionsTime = sessionsTime + time;
  }

 /**
   * Increment the number of succesfully ended sessions.
   */
  public synchronized void addSession()
  {
    nbSessions++;
  }


  /**
   * Increment an entry count by one.
   *
   * @param index index of the entry
   */
  public synchronized void incrementCount(int index)
  {
    count[index]++;

    int session_num = nbSessions;
    String filename_session = savePath + "/session.txt";
    try {
      record(0, filename_session, "" + session_num, true);
    } catch (Exception e) {
      e.printStackTrace();
    }
    


  }


  /**
   * Increment an entry error by one.
   *
   * @param index index of the entry
   */
  public synchronized void incrementError(int index)
  {
    error[index]++;
  }


  /**
   * Add a new time sample for this entry. <code>time</code> is added to total time
   * and both minTime and maxTime are updated if needed.
   *
   * @param index index of the entry
   * @param time time to add to this entry
   */
  public synchronized void updateTime(int index, long time)
  { //time: Response time
    if (time < 0)
    {
      System.err.println("Negative time received in Stats.updateTime("+time+")<br>\n");
      return ;
    }

    int num = getCount(index);
    int error_num = getError(index);
    num = num + error_num;
    int rt = (int)time;
    String filename = savePath + "/" + index + "-RT-count.txt";

    try {
      record(index, filename, rt + " " + num,  true);
    } catch (Exception e) {
      e.printStackTrace();
    }

    totalTime[index] += time;
    if (time > maxTime[index])
      maxTime[index] = time;
    if (time < minTime[index])
      minTime[index] = time;
  }


  /**
   * Get current count of an entry
   *
   * @param index index of the entry
   *
   * @return entry count value
   */
  public synchronized int getCount(int index)
  {
    return count[index];
  }


  /**
   * Get current error count of an entry
   *
   * @param index index of the entry
   *
   * @return entry error value
   */
  public synchronized int getError(int index)
  {
    return error[index];
  }


  /**
   * Get the minimum time of an entry
   *
   * @param index index of the entry
   *
   * @return entry minimum time
   */
  public synchronized long getMinTime(int index)
  {
    return minTime[index];
  }


  /**
   * Get the maximum time of an entry
   *
   * @param index index of the entry
   *
   * @return entry maximum time
   */
  public synchronized long getMaxTime(int index)
  {
    return maxTime[index];
  }


  /**
   * Get the total time of an entry
   *
   * @param index index of the entry
   *
   * @return entry total time
   */
  public synchronized long getTotalTime(int index)
  {
    return totalTime[index];
  }


  /**
   * Get the total number of entries that are collected
   *
   * @return total number of entries
   */
  public int getNbOfStats()
  {
    return nbOfStats;
  }


  /**
   * Adds the entries of another Stats object to this one.
   *
   * @param anotherStat stat to merge with current stat
   */
  public synchronized void merge(Stats anotherStat)
  {
    if (this == anotherStat)
    {
      System.out.println("You cannot merge a stats with itself");
      return;
    }
    if (nbOfStats != anotherStat.getNbOfStats())
    {
      System.out.println("Cannot merge stats of differents sizes.");
      return;
    }
    for (int i = 0 ; i < nbOfStats ; i++)
    {
      count[i] += anotherStat.getCount(i);
      error[i] += anotherStat.getError(i);
      if (minTime[i] > anotherStat.getMinTime(i))
        minTime[i] = anotherStat.getMinTime(i);
      if (maxTime[i] < anotherStat.getMaxTime(i))
        maxTime[i] = anotherStat.getMaxTime(i);
      totalTime[i] += anotherStat.getTotalTime(i);
    }
    nbSessions   += anotherStat.nbSessions;
    sessionsTime += anotherStat.sessionsTime;
  }


  /**
   * Display an HTML table containing the stats for each state.
   * Also compute the totals and average throughput
   *
   * @param title table title
   * @param sessionTime total time for this session
   * @param exclude0Stat true if you want to exclude the stat with a 0 value from the output
   */
  public void display_stats(String title, long sessionTime, boolean exclude0Stat)
  {
    int counts = 0;
    int errors = 0;
    long time = 0;

    System.out.println("<br><h3>"+title+" statistics</h3><p>");
    System.out.println("<TABLE BORDER=1>");
    System.out.println("<THEAD><TR><TH>State name<TH>% of total<TH>Count<TH>Errors<TH>Minimum Time<TH>Maximum Time<TH>Average Time<TBODY>");
    // Display stat for each state
    for (int i = 0 ; i < getNbOfStats() ; i++)
    {
      counts += count[i];
      errors += error[i];
      time += totalTime[i];
    }

    for (int i = 0 ; i < getNbOfStats() ; i++)
    {
      if ((exclude0Stat && count[i] != 0) || (!exclude0Stat))
      {
        System.out.print("<TR><TD><div align=left>"+TransitionTable.getStateName(i)+"</div><TD><div align=right>");
        if ((counts > 0) && (count[i] > 0))
          System.out.print(100*count[i]/counts+" %");
        else
          System.out.print("0 %");
        System.out.print("</div><TD><div align=right>"+count[i]+"</div><TD><div align=right>");
        if (error[i] > 0)
          System.out.print("<B>"+error[i]+"</B>");
        else
          System.out.print(error[i]);
        System.out.print("</div><TD><div align=right>");
        if (minTime[i] != Long.MAX_VALUE)
          System.out.print(minTime[i]);
        else
          System.out.print("0");
        System.out.print(" ms</div><TD><div align=right>"+maxTime[i]+" ms</div><TD><div align=right>");
        if (count[i] != 0)
          System.out.println(totalTime[i]/count[i]+" ms</div>");
        else
           System.out.println("0 ms</div>");
      }
    }

    // Display total   
    if (counts > 0)
    {
      System.out.print("<TR><TD><div align=left><B>Total</B></div><TD><div align=right><B>100 %</B></div><TD><div align=right><B>"+counts+
                       "</B></div><TD><div align=right><B>"+errors+ "</B></div><TD><div align=center>-</div><TD><div align=center>-</div><TD><div align=right><B>");
      counts += errors;
      System.out.println(time/counts+" ms</B></div>");
      // Display stats about sessions
      System.out.println("<TR><TD><div align=left><B>Average throughput</div></B><TD colspan=6><div align=center><B>"+1000*counts/sessionTime+" req/s</B></div>");
      System.out.println("<TR><TD><div align=left>Completed sessions</div><TD colspan=6><div align=left>"+nbSessions+"</div>");
      System.out.println("<TR><TD><div align=left>Total time</div><TD colspan=6><div align=left>"+sessionsTime/1000L+" seconds</div>");
      System.out.print("<TR><TD><div align=left><B>Average session time</div></B><TD colspan=6><div align=left><B>");
      if (nbSessions > 0)
        System.out.print(sessionsTime/(long)nbSessions/1000L+" seconds");
      else
        System.out.print("0 second");
      System.out.println("</B></div>");
      //String filename_session = "/home/wxh/data/sampleFromRubis-session.txt";
      //String filename_RT = "/home/wxh/data/sampleFromRubis-RT.txt";
      //add: Record number of session and response time in real-time.

    }
    System.out.println("</TABLE><p>");
    
  }

  public void record(int index, String filename, String counter, boolean append_flag) throws Exception{
        
    Calendar c = Calendar.getInstance();
    SimpleDateFormat forr = new SimpleDateFormat("HH:mm:ss");
    String stamp_r = forr.format(c.getTime());
    if (stamp_r.equals(time_now[index]) == false){
      time_now[index]= stamp_r;
      File file = new File(filename);
      if (file.exists() == false){
          file.createNewFile();
      }
      if (append_flag){
        record_a(filename, counter);
      }
      else{
        FileOutputStream fos = new FileOutputStream(filename);
      
        SimpleDateFormat formatter = new SimpleDateFormat("HH:mm:ss");
        
        String stamp = formatter.format(c.getTime());
        String outString = stamp + " " + counter;
        byte output[] = outString.getBytes("UTF-8");

        fos.write(output);
        fos.close();
      }
    }
    
    
  }

  public void record_a(String fileName, String content) {
    try {
    // 打开一个随机访问文件流，按读写方式
    Calendar c = Calendar.getInstance();
    SimpleDateFormat formatter = new SimpleDateFormat("HH:mm:ss");
    String stamp = formatter.format(c.getTime());

    RandomAccessFile randomFile = new RandomAccessFile(fileName, "rw");
    // 文件长度，字节数
    long fileLength = randomFile.length();
    // 将写文件指针移到文件尾。
    randomFile.seek(fileLength);
    randomFile.writeBytes(stamp + " " + content+"\n");
    randomFile.close();
    } catch (IOException e) {
    e.printStackTrace();
    }
    }

    public Long sortFileByName(String PATH, final String orderStr) {
      File pth = new File(PATH);
      File[] files = pth.listFiles();
      if (!orderStr.equalsIgnoreCase("asc") && orderStr.equalsIgnoreCase("desc")) {
          long a = 99;
          return a;
      }
      List<File> tmp = new ArrayList<File>();
      for(File f: files){
          if (f.isDirectory() == true){
              tmp.add(f);
              
          }
      }
      File[] files1 = tmp.toArray(new File[0]);
      
      Arrays.sort(files1, new Comparator<File>() {
          public int compare(File o1, File o2) {
              int n1 = extractNumber(o1.getName());
              int n2 = extractNumber(o2.getName());
              if(orderStr == null || orderStr.length() < 1 || orderStr.equalsIgnoreCase("asc")) {
                  return n1 - n2;
              } else {
                  //降序
                  return n2 - n1;
              }
          }
      });
      Date date = new Date();
      long now = date.getTime();
      long last = new Date(files1[files1.length - 1].lastModified()).getTime();

      return (now - last) / 1000;
  }

  private int extractNumber(String name) {
      int i;
      try {
          String number = name.replaceAll("[^\\d]", "");
          i = Integer.parseInt(number);
      } catch (Exception e) {
          i = 0;
      }
      return i;
  }

}
