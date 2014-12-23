package ch.eonum.pipeline.clustering;

import java.util.HashMap;
import java.util.Map;


import ch.eonum.pipeline.util.Log;
import ch.eonum.pipeline.core.DataSet;
import ch.eonum.pipeline.core.Features;
import ch.eonum.pipeline.core.Instance;


/**
 * KMeans clustering. @link http://en.wikipedia.org/wiki/K-means_clustering
 * 
 * @author tim
 *
 */
public class KmeansClustering<E extends Instance> extends Clustering<E> {
	protected static final Map<String, String> PARAMETERS = new HashMap<String, String>();
	
	static {
		PARAMETERS.put("k", "number of clusters " +
				":  (default 5.0)");
		PARAMETERS.put("maxIter", "number of iterations " +
				":  (default 15.0)");
	}
	
	/** variance of each cluster. */
	private HashMap<String, Instance> clusterVariances;

	public KmeansClustering(){
		super();
		this.setSupportedParameters(KmeansClustering.PARAMETERS);
		this.putParameter("k", 5.0);
		this.putParameter("maxIter", 10.0);
	}

	/**
	 * @param dataset1 cluster assignments 
	 * @param dataset2 data
	 */
	public KmeansClustering(DataSet<E> data, Features features) {
		super();
		this.setFeatures(features);
		calculateClusters(data);
	}	

	public KmeansClustering(String folderName) {
		super(folderName);
	}

	public void calculateClusters(DataSet<E> data) {
		this.clusterCenters = new HashMap<String, Instance>();
		this.normalizedClusterCenters = new HashMap<String, Instance>();
		this.clusterSizes = new HashMap<String, Integer>();
		this.clusterLengths = new HashMap<String, Double>();
		
		/**
		 * Alternatively initialization could be done using a prototype
		 * selection algorithm. E.g. Spanning Prototype Selection.
		 */
		this.initRandom(data);
		
		for(int iteration = 0; iteration < (int)this.getDoubleParameter("maxIter"); iteration++){
			for(String cluster : this.clusterCenters.keySet()){
				this.clusterSizes.put(cluster, 1);
			}
			Map<String, Instance> clusterCentersTemp = new HashMap<String, Instance>();
			Log.puts("KMeans Iteration " + iteration);
			for(Instance each : data){
				String cluster = this.getMaxCluster(each);
				if(clusterCentersTemp.containsKey(cluster)){
					clusterCentersTemp.get(cluster).add(each);
					this.clusterSizes.put(cluster, clusterSizes.get(cluster) + 1); 
				}
				else {
					clusterCentersTemp.put(cluster, each.copy());
					this.clusterSizes.put(cluster, 1); 
				}	
				
			}
			for(String e : clusterCentersTemp.keySet())
				this.clusterCenters.put(e, clusterCentersTemp.get(e));
			for(String each : clusterCenters.keySet()){
				clusterCenters.get(each).divideBy(clusterSizes.get(each));
				Log.puts("" + clusterSizes.get(each));
			}
			for(Instance each : clusterCenters.values()){
				double length = this.distance.length(each);
				this.clusterLengths.put(each.id, length);
				Instance normedCenter = each.copy();
				normedCenter.divideBy(length);
				this.normalizedClusterCenters.put(each.id, normedCenter);
			}
		}
		
		Log.puts("Calculating variances");
		this.clusterVariances = new HashMap<String, Instance>();
		for(Instance each : data){
			String cluster = this.getMaxCluster(each);
			Instance diff = each.minusStateless(this.clusterCenters.get(cluster));
			diff = diff.timesStateless(diff);
			if(clusterVariances.containsKey(cluster))
				clusterVariances.get(cluster).add(diff);
			else
				clusterVariances.put(cluster, diff);
		}
		for(String each : clusterVariances.keySet()){
			clusterVariances.get(each).divideBy(clusterSizes.get(each));
		}
	}

	/**
	 * return the nearest cluster for an instance
	 * @param inst
	 * @return
	 */
	private String getMaxCluster(Instance inst) {
		String maxInst = null;
		double nearest = Double.POSITIVE_INFINITY;
		for(String clusterCenter : this.clusterCenters.keySet()){
			double distance = this.distance.distance(this.clusterCenters.get(clusterCenter), inst);
			if(distance < nearest){
				nearest = distance;
				maxInst = clusterCenter;
			}
		}
		return maxInst;
	}

	private void initRandom(DataSet<E> data) {
		int k = (int)this.getDoubleParameter("k");
		Log.puts("Number of clusters: " + k);
		for(int i = 0; i < k; i++){
			String clusterName = String.valueOf(i);
			Instance inst = data.get((int)(Math.random()*(data.size()-1)));
			this.clusterCenters.put(clusterName, inst);
		}
	}

	@Override
	public void train() {
		this.calculateClusters(this.trainingDataSet);
	}
	
	@Override
	public DataSet<E> test() {		
		for(E inst : this.testDataSet)
			inst.putResult("result", this.distance(inst.className, inst));
		return testDataSet;
	}

	public Map<String, Instance> getCovariances() {
		return this.clusterVariances;
	}

}
