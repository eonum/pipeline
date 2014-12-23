package ch.eonum.pipeline.clustering;

import java.util.HashMap;
import java.util.List;


import ch.eonum.pipeline.core.DataSet;
import ch.eonum.pipeline.core.Instance;
import ch.eonum.pipeline.core.SparseInstance;

/**
 * Assign instances to clusters according to an external grouping.
 * 
 * A static clustering is generated of a data set.
 * the group/cluster name has to be set for each instance in the data set.
 * 
 * The data set is not changed.
 * 
 * @author tim
 *
 */
public class StaticClustering<E extends Instance> extends Clustering<E> {
	
	
	public StaticClustering(){
		super();
	}

	/**
	 * 
	 * @param dataset1 cluster assignments 
	 * @param dataset2 data
	 */
	public StaticClustering(DataSet<E> data) {
		super();
		calculateClusters(data);
	}	

	public StaticClustering(String folderName) {
		super(folderName);
	}

	public void calculateClusters(DataSet<E> data) {
		this.clusterCenters = new HashMap<String, Instance>();
		this.normalizedClusterCenters = new HashMap<String, Instance>();
		this.clusterSizes = new HashMap<String, Integer>();
		this.clusterLengths = new HashMap<String, Double>();
		
		for(Instance each : data){
			if(!clusterCenters.containsKey(each.className)){
				clusterCenters.put(each.className, new SparseInstance(each.className, each.groundTruth, new HashMap<String, Double>()));
				clusterSizes.put(each.className, 0);
			}
			Instance cluster = clusterCenters.get(each.className);
			cluster.add(each);
			this.clusterSizes.put(cluster.id, clusterSizes.get(cluster.id) + 1); 
		}
		
		for(Instance each : clusterCenters.values())
			each.divideBy(clusterSizes.get(each.id));
		
		
		for(Instance each : clusterCenters.values()){
			double length = this.distance.length(each);
			this.clusterLengths.put(each.id, length);
			Instance normedCenter = each.copy();
			normedCenter.divideBy(length);
			this.normalizedClusterCenters.put(each.id, normedCenter);
		}
	}

	@Override
	public void train() {
		this.calculateClusters(this.trainingDataSet);
	}
	
	@Override
	public DataSet<E> test() {		
		for(Instance inst : this.testDataSet){
			inst.putResult("result", this.distance(inst.className, inst));
			List<ClusterComparison> ranking = this.getRanking(inst);
			inst.label = ranking.get(0).getCluster();
		}
		return testDataSet;
	}

}
