using System;
using System.IO;
using System.IO.Compression;
using System.Text;
using System.Text.RegularExpressions;
using System.Collections.Generic;

public static class SemanticSensitivity
{
    static Dictionary<string,object> Row(params object[] pairs) {
        var d=new Dictionary<string,object>();
        for(int i=0;i<pairs.Length;i+=2)d.Add((string)pairs[i],pairs[i+1]);
        return d;
    }
    static object Rate(int n,int d) { return d==0?null:(object)((double)n/d); }
    static bool Pass(byte[] a,int m,int r,int c,int nr,int nc,int bit) {
        return (a[(m*nr+r)*nc+c]&bit)!=0;
    }
    static bool All(byte[] a,int m,int r,int first,int end,int nr,int nc,int bit) {
        for(int c=first;c<end;c++)if(!Pass(a,m,r,c,nr,nc,bit))return false;
        return true;
    }
    static int Images(bool[] selected,string[] images) {
        var h=new HashSet<string>();for(int r=0;r<images.Length;r++)if(selected[r])h.Add(images[r]);return h.Count;
    }
    public static Dictionary<string,object> Analyze(byte[] flags,string[] models,string[] roots,string[] images,int singles,int[] imageAnchors,int[] textAnchors,bool[] diagonal,bool[] corner) {
        int nm=models.Length,nr=roots.Length,nj=diagonal.Length,nc=1+singles+nj,first=1+singles;
        if(nm<1||nr<1||images.Length!=nr||flags.Length!=nm*nr*nc||corner.Length!=nj||imageAnchors.Length!=nj||textAnchors.Length!=nj)throw new Exception("Input dimensions mismatch");
        if(new HashSet<string>(models).Count!=nm||new HashSet<string>(roots).Count!=nr)throw new Exception("Duplicate identifiers");
        for(int j=0;j<nj;j++)if(imageAnchors[j]<1||imageAnchors[j]>singles||textAnchors[j]<1||textAnchors[j]>singles)throw new Exception("Invalid corresponding anchor");
        foreach(byte b in flags)if(b<8||b>15||((b&4)!=0)!=(((b&1)!=0)&&((b&2)!=0)))throw new Exception("Incomplete or inconsistent task conjunction");
        var summaries=new List<Dictionary<string,object>>();var rootRows=new List<Dictionary<string,object>>();
        var fixedRows=new List<Dictionary<string,object>>();var replacements=new List<Dictionary<string,object>>();
        int[] bits={4,1};string[] axes={"task_pass","semantic_correct"};
        for(int ax=0;ax<bits.Length;ax++) {
            int bit=bits[ax];
            for(int m=0;m<nm;m++) {
                int clean=0,single=0,joint=0,q=0,affected=0,events=0,df=0,cf=0,full=0,eligible=0,gaps=0;
                var qMask=new bool[nr];var affectedMask=new bool[nr];var eventSizes=new List<int>();
                for(int r=0;r<nr;r++) {
                    bool qualified=All(flags,m,r,0,first,nr,nc,bit),suite=All(flags,m,r,0,nc,nr,nc,bit),dBad=false,cBad=false;
                    int failures=0,er=0,gr=0;
                    if(Pass(flags,m,r,0,nr,nc,bit))clean++;
                    for(int c=1;c<first;c++)if(Pass(flags,m,r,c,nr,nc,bit))single++;
                    for(int j=0;j<nj;j++) {
                        bool pass=Pass(flags,m,r,first+j,nr,nc,bit);
                        if(pass)joint++;else {failures++;if(diagonal[j])dBad=true;if(corner[j])cBad=true;}
                        if(Pass(flags,m,r,imageAnchors[j],nr,nc,bit)&&Pass(flags,m,r,textAnchors[j],nr,nc,bit)){er++;if(!pass)gr++;}
                    }
                    eligible+=er;gaps+=gr;if(suite)full++;
                    if(qualified){q++;qMask[r]=true;events+=failures;if(failures>0){affected++;affectedMask[r]=true;eventSizes.Add(failures);}if(dBad)df++;if(cBad)cf++;}
                    rootRows.Add(Row("model",models[m],"axis",axes[ax],"root",roots[r],"image",images[r],"qualified",qualified,"all_suite_pass",suite,"joint_failures",failures,"diagonal_failed",dBad,"corner_failed",cBad,"cell_eligible",er,"cell_gaps",gr));
                }
                eventSizes.Sort();object median=null;if(eventSizes.Count>0){int k=eventSizes.Count/2;median=eventSizes.Count%2==1?(double)eventSizes[k]:(eventSizes[k-1]+eventSizes[k])/2.0;}
                summaries.Add(Row("model",models[m],"axis",axes[ax],"roots",nr,"images",new HashSet<string>(images).Count,"clean_correct",clean,"single_correct",single,"single_denominator",nr*singles,"joint_correct",joint,"joint_denominator",nr*nj,"qualified_roots",q,"qualified_images",Images(qMask,images),"affected_roots",affected,"affected_images",Images(affectedMask,images),"affected_fraction",Rate(affected,q),"failure_events",events,"event_denominator",q*nj,"event_density",Rate(events,q*nj),"median_events_affected",median,"diagonal_found",df,"diagonal_missed",affected-df,"corner_found",cf,"corner_missed",affected-cf,"all_suite_pass",full,"cell_eligible",eligible,"cell_gaps",gaps,"cell_gap_rate",Rate(gaps,eligible)));
            }
        }
        // Keep primary eligibility fixed: a separate answer from semantic requalification.
        for(int m=0;m<nm;m++) {
            int q=0,ta=0,sa=0,te=0,se=0,formatOnly=0,df=0,cf=0;
            for(int r=0;r<nr;r++)if(All(flags,m,r,0,first,nr,nc,4)) {
                q++;bool taskBad=false,semBad=false,dBad=false,cBad=false;
                for(int j=0;j<nj;j++) {
                    if(!Pass(flags,m,r,first+j,nr,nc,4)){taskBad=true;te++;}
                    if(!Pass(flags,m,r,first+j,nr,nc,1)){semBad=true;se++;if(diagonal[j])dBad=true;if(corner[j])cBad=true;}
                }
                if(taskBad)ta++;if(semBad)sa++;if(taskBad&&!semBad)formatOnly++;if(dBad)df++;if(cBad)cf++;
            }
            fixedRows.Add(Row("model",models[m],"cohort","primary_task_strict41","qualified_roots",q,"task_affected_roots",ta,"semantic_affected_roots",sa,"format_only_affected_roots",formatOnly,"task_failure_events",te,"semantic_failure_events",se,"format_only_failure_events",te-se,"semantic_diagonal_found",df,"semantic_diagonal_missed",sa-df,"semantic_corner_found",cf,"semantic_corner_missed",sa-cf));
        }
        for(int ax=0;ax<bits.Length;ax++)for(int old=0;old<nm;old++)for(int target=0;target<nm;target++)if(old!=target) {
            int bit=bits[ax];
            foreach(string cohort in new string[]{"reference_full_suite","reference_strict41","common_strict41"}) {
                int n=0,cr=0,sr=0,jr=0,p=0,repaired=0,persistent=0,rc=0,ncBad=0;
                for(int r=0;r<nr;r++) {
                    bool oq=All(flags,old,r,0,first,nr,nc,bit),tq=All(flags,target,r,0,first,nr,nc,bit);
                    bool os=All(flags,old,r,0,nc,nr,nc,bit);
                    bool use=cohort=="reference_full_suite"?os:cohort=="reference_strict41"?oq:oq&&tq;
                    if(!use)continue;n++;
                    bool oldJoint=All(flags,old,r,first,nc,nr,nc,bit),newJoint=All(flags,target,r,first,nc,nr,nc,bit);
                    if(cohort=="common_strict41") {
                        if(oldJoint&&!newJoint)jr++;if(oldJoint&&newJoint)p++;
                        if(!oldJoint&&newJoint)repaired++;if(!oldJoint&&!newJoint)persistent++;
                        for(int j=0;j<nj;j++){
                            bool op=Pass(flags,old,r,first+j,nr,nc,bit),tp=Pass(flags,target,r,first+j,nr,nc,bit);
                            if(!op&&tp)rc++;if(op&&!tp)ncBad++;
                        }
                    } else {
                        if(!Pass(flags,target,r,0,nr,nc,bit))cr++;
                        else if(!tq)sr++;
                        else if(!newJoint)jr++;else p++;
                    }
                }
                if((cohort=="common_strict41"?jr+p+repaired+persistent:cr+sr+jr+p)!=n)throw new Exception("Replacement stage partition mismatch");
                replacements.Add(Row("axis",axes[ax],"reference",models[old],"target",models[target],"cohort",cohort,"denominator",n,"clean_regression",cr,"single_regression",sr,"joint_regression",jr,"preserved",p,"repaired_roots",cohort=="common_strict41"?(object)repaired:null,"persistent_roots",cohort=="common_strict41"?(object)persistent:null,"repaired_conditions",cohort=="common_strict41"?(object)rc:null,"new_failed_conditions",cohort=="common_strict41"?(object)ncBad:null));
            }
        }
        // Semantic target outcomes on the original task all-suite reference cohort.
        for(int old=0;old<nm;old++)for(int target=0;target<nm;target++)if(old!=target) {
            int n=0,cr=0,sr=0,jr=0,p=0;
            for(int r=0;r<nr;r++)if(All(flags,old,r,0,nc,nr,nc,4)){
                n++;if(!Pass(flags,target,r,0,nr,nc,1))cr++;
                else if(!All(flags,target,r,1,first,nr,nc,1))sr++;
                else if(!All(flags,target,r,first,nc,nr,nc,1))jr++;else p++;
            }
            replacements.Add(Row("axis","semantic_correct","reference",models[old],"target",models[target],"cohort","fixed_primary_reference_full_suite","denominator",n,"clean_regression",cr,"single_regression",sr,"joint_regression",jr,"preserved",p,"repaired_roots",null,"persistent_roots",null,"repaired_conditions",null,"new_failed_conditions",null));
        }
        return Row("summaries",summaries,"fixed_primary_cohorts",fixedRows,"replacements",replacements,"root_records",rootRows);
    }
    public class Npy { public string Descriptor;public int[] Shape;public byte[] Data; }
    public static Npy ReadNpy(string archive,string member) {
        using(var zip=ZipFile.OpenRead(archive)) {
            var entry=zip.GetEntry(member);if(entry==null)throw new Exception("Missing NPY member");
            using(var stream=entry.Open())using(var reader=new BinaryReader(stream)) {
                byte[] magic=reader.ReadBytes(6);if(magic.Length!=6||magic[0]!=147||Encoding.ASCII.GetString(magic,1,5)!="NUMPY")throw new Exception("Invalid NPY magic");
                int major=reader.ReadByte();reader.ReadByte();int length=major==1?reader.ReadUInt16():major==2?checked((int)reader.ReadUInt32()):throw new Exception("Unsupported NPY version");
                string header=Encoding.ASCII.GetString(reader.ReadBytes(length));
                var dtype=Regex.Match(header,@"'descr':\s*'([^']+)'");var shape=Regex.Match(header,@"'shape':\s*\(([^)]+)\)");
                if(!dtype.Success||!shape.Success||!header.Contains("'fortran_order': False"))throw new Exception("Unsupported NPY header");
                var dims=new List<int>();foreach(Match d in Regex.Matches(shape.Groups[1].Value,@"\d+"))dims.Add(int.Parse(d.Value));
                using(var ms=new MemoryStream()){stream.CopyTo(ms);return new Npy{Descriptor=dtype.Groups[1].Value,Shape=dims.ToArray(),Data=ms.ToArray()};}
            }
        }
    }
    public static string[] Strings(string archive,string member) {
        var a=ReadNpy(archive,member);if(a.Shape.Length!=1||!Regex.IsMatch(a.Descriptor,@"^<U\d+$"))throw new Exception("Unsupported Unicode array");
        int stride=int.Parse(a.Descriptor.Substring(2))*4;if(a.Data.Length!=a.Shape[0]*stride)throw new Exception("Unicode byte count");
        var strings=new string[a.Shape[0]];for(int i=0;i<strings.Length;i++)strings[i]=Encoding.UTF32.GetString(a.Data,i*stride,stride).TrimEnd('\0');return strings;
    }
}
