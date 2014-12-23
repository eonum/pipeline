package ch.eonum.pipeline.clustering;

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ch.eonum.pipeline.classification.Classifier;
import ch.eonum.pipeline.core.Instance;
import ch.eonum.pipeline.core.SparseInstance;
import ch.eonum.pipeline.distance.Distance;
import ch.eonum.pipeline.distance.MinkowskiDistance;
import ch.eonum.pipeline.util.FileUtil;

/**
 * Abstract clustering class. Provides a metric/distance, clusters and cluster
 * serialization/deserialization.
 * 
 * 
 * @author tim
 * 
 */
public abstract class Clustering<E extends Instance> extends Classifier<E> {
	protected static final Map<String, String> PARAMETERS = new HashMap<String, String>();
	protected Distance<Instance> distance;
	
	static {
		PARAMETERS.put("minkowski", "minkowski constant for the minkowski metric, 2.0 => euclidian metric" +
				":  (default 2.0)");
		PARAMETERS.put("exponent", "exponent for the polynomial kernel" +
		":  (default 1.0)");
	}
	
	/** unnormalized cluster centers. */
	protected Map<String, Instance> clusterCenters;
	/** normalized cluster centers. */
	protected Map<String, Instance> normalizedClusterCenters;
	/** number of training samples for each cluster. */
	protected Map<String, Integer> clusterSizes;
	/**
	 * distance to the origin for each cluster. used as cache for optimized
	 * distance calculations.
	 */
	protected HashMap<String, Double> clusterLengths;

	public Clustering() {
		super();
		this.setSupportedParameters(StaticClustering.PARAMETERS);
		this.distance = new MinkowskiDistance<Instance>(2.0);
		this.putParameter("minkowski", 2.0);
		this.putParameter("exponent", 2.0);
	}
	
	/**
	 * read clusters from folder.
	 * 
	 * @param folderName
	 */
	public Clustering(String folderName) {
		this();
		loadFromFolder(folderName);
	}
	
	@Override
	public void putParameter(String p, double d){
		super.putParameter(p, d);
		if(this.distance.getSupportedParameters().containsKey(p))
			this.distance.putParameter(p, d);
	}

	/**
	 * deserialize.
	 * @param folderName
	 */
	private void loadFromFolder(String folderName) {
		this.clusterCenters = new HashMap<String, Instance>();
		this.normalizedClusterCenters = new HashMap<String, Instance>();
		this.clusterSizes = new HashMap<String, Integer>();
		this.clusterLengths = new HashMap<String, Double>();
		
		File folder = new File(folderName);
		
		try {
			for (String each : folder.list()) {
				FileInputStream fstream = new FileInputStream(folder.getAbsolutePath() + File.separator + each);
				DataInputStream in = new DataInputStream(fstream);
				BufferedReader br = new BufferedReader(
						new InputStreamReader(in));
				String strLine;
				String cluster = each.replace(".txt", "");
				Map<String, Double> data = new HashMap<String, Double>();
				/** the first line is the size:. */
				this.clusterSizes.put(cluster, Integer.valueOf(br.readLine()));
				while ((strLine = br.readLine()) != null) {
					String[] values = strLine.split(" ");
					data.put(values[0], Double.valueOf(values[1]));
				}
				Instance inst = new SparseInstance(cluster, cluster, data);
				this.clusterCenters.put(cluster, inst);
				in.close();
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		for(Instance each : clusterCenters.values()){
			double length = this.distance.length(each);
			this.clusterLengths.put(each.id, length);
			Instance normedCenter = each.copy();
			normedCenter.divideBy(length);
			this.normalizedClusterCenters.put(each.id, normedCenter);
		}
	}
	
	public Map<String, Instance> getClusters() {
		return this.clusterCenters;
	}

	public Map<String, Integer> getClusterSizes() {
		return this.clusterSizes;
	}

	public HashMap<String, Double> getClusterLengths() {
		return this.clusterLengths;
	}

	/**
	 * calculate the distance between a cluster and an instance. the distance is
	 * calculated on the normalized cluster center and the normalized instance.
	 * the resulting distance is between 0 and 1;
	 * 
	 * @param clusterName
	 * @param instance
	 */
	public double distance(String clusterName, Instance inst) {
		Instance normedInst = inst.copy();
		normedInst.divideBy(this.distance.length(inst));
		
		Instance cluster = normalizedClusterCenters.get(clusterName);
		if(cluster == null)
			cluster = new SparseInstance(clusterName, "0", new HashMap<String, Double>());
		
		return this.distance.distance(normedInst, cluster, 1.0);
	}

	/**
	 * write all cluster centers to files. The unnormalized centers are written
	 * to file.
	 * 
	 * @param folderName
	 * @throws IOException 
	 */
	public void writeToFile(String folderName) throws IOException {
		File folder = new File(folderName);
		folder.mkdir();

		for (Instance cluster : this.clusterCenters.values()) {
			PrintStream p = new PrintStream(new FileOutputStream(
					folder.getAbsolutePath() + File.separator + cluster.id
							+ ".txt"));
			p.println(this.clusterSizes.get(cluster.id));
			for (String feature : cluster.features())
				p.println(feature + " " + cluster.get(feature));
			p.close();
		}
	}
	
	@Override
	public void save(String fileName) throws IOException {
		super.save(fileName);
		FileUtil.mkdir(fileName + "_clusters/");
		this.writeToFile(fileName + "_clusters/");
	}
	
	@Override
	public void loadSerializedState(File file) throws IOException {
		super.loadSerializedState(file);
		this.loadFromFolder(file.getAbsolutePath() + "_clusters/");
	}

	/**
	 * Get a list of all clusters and their distance ordered by their distance
	 * to the provided instance in ascending order.
	 * 
	 * @param inst
	 * @return ranking
	 */
	public List<ClusterComparison> getRanking(Instance inst) {
		List<ClusterComparison> comps = new ArrayList<ClusterComparison>();
		for(String cluster : this.clusterCenters.keySet()){
			comps.add(new ClusterComparison(cluster, this.distance(cluster, inst)));
		}
		Collections.sort(comps);
		return comps;	
	}
	
	public class ClusterComparison implements Comparable<ClusterComparison> {
		private String cluster;
		private double distance;
		public ClusterComparison(String cluster, double distance){
			this.cluster = cluster;
			this.distance = distance;
		}
		
		public String getCluster() {
			return cluster;
		}
		
		public double getDistance() {
			return distance;
		}

		@Override
		public int compareTo(ClusterComparison other) {
			return ((Double)distance).compareTo(other.getDistance());
		}
	}

}