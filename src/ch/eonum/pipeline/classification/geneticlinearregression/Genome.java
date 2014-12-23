package ch.eonum.pipeline.classification.geneticlinearregression;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import Jama.Matrix;

import ch.eonum.pipeline.core.Features;

/**
 * A genome, one possible solution for the problem which has to prove itself
 * against all other genomes in the population using the fitness function.
 * 
 * @author tim
 * 
 */
public class Genome implements Comparable<Genome> {

	/** deterministic random number generator. */
	private Random random;
	/** The actual algorithm. phenotype. */
	private List<Node> nodes;
	/**
	 * Introns. Nodes that do not currently have a function. In later
	 * generations an intron can become a functional node.
	 */
	private List<Node> introns;
	/**
	 * meta parameters such as crossover and mutation probability. part of the
	 * genotype but not of the phenotype.
	 */
	private Map<String, Double> parameters;
	/** linear regression matrix. */
	private Matrix beta;
	/** fitness as calculated by a fitness function. determines survival. */
	private double fitness;
	/** global max nodes. */
	private int maxNodes;
	public final static String[] paramTypes = { "crossover",
			"crossover-intern", "mutation", "removal", "replacement", "crossover-add" };

	/**
	 * Create a random genome.
	 * @param features
	 * @param seed
	 */
	public Genome(Features features, int seed, int maxNodes, int maxDepth) {
		this.random = new Random(seed);
		parameters = new HashMap<String, Double>();
		for(String param : paramTypes)
			parameters.put(param, Math.min(0.9, Math.max(0.1, random.nextDouble())));
		maxDepth = maxDepth == 0 ? 0 : random.nextInt(maxDepth) + 1;
		int numNodes = random.nextInt(maxNodes);
		this.maxNodes = maxNodes;
		this.nodes = new ArrayList<Node>();
		double endNodesProb = random.nextDouble();
		double timesNodesProb = random.nextDouble();
		double minusNodes = random.nextDouble();
		double thresholdNodesProb = random.nextDouble();
		double total = endNodesProb + timesNodesProb + minusNodes + thresholdNodesProb;
		endNodesProb /= total;
		timesNodesProb /= total;
		thresholdNodesProb /= total;
		for (int i = 0; i < numNodes; i++) {
			double nodeType = random.nextDouble();
			nodes.add(Node.randomNode(endNodesProb, timesNodesProb, thresholdNodesProb, nodeType,
					features, random, 0, maxDepth));
		}
	}
	
	public Genome(int seed) {
		this.random = new Random(seed);
		this.nodes = new ArrayList<Node>();
	}

	@Override
	public String toString(){
		String genome = "Genome size: " + nodes.size();
		for(Node node : nodes)
			genome +=  "\n" + node;
		return genome;
	}

	/**
	 * Transform a data set and set the first row to 1.0
	 * 
	 * @param trainMatrix
	 * @return
	 */
	public double[][] transform(double[][] trainMatrix) {
		double[][] transformed = new double[trainMatrix.length][nodes.size()+1];
		for(int i = 0; i < trainMatrix.length; i++){
			transformed[i][0] = 1.0;
			for(int n = 0; n < nodes.size(); n++)
				transformed[i][n+1] = nodes.get(n).evaluate(trainMatrix[i]);
		}
		return transformed;
	}

	public int size() {
		return this.nodes.size();
	}

	/**
	 * remove all introns. An intron is a node, which evaluates to the same
	 * value for all training data instances or is a duplicate of another node.
	 * 
	 * @param trainMatrix
	 */
	public void removeIntrons(double[][] trainMatrix) {
		/** duplicates. */
		this.introns = new ArrayList<Node>();
		for (Node node : nodes)
			if (!introns.contains(node))
				for (Node node2 : nodes)
					if (!introns.contains(node2))
						if(node != node2)
							if (node.toString().equals(node2.toString()))
								introns.add(node2);
		/** constants. */
		for(int n = 0; n < nodes.size(); n++){
			if(introns.contains(nodes.get(n))) continue;
			double value = nodes.get(n).evaluate(trainMatrix[0]);
			boolean intron = true;
			for(int i = 0; i < trainMatrix.length; i++){
				if(Math.abs(value - nodes.get(n).evaluate(trainMatrix[i])) > 0.00001){
					intron = false;
					break;
				}
			}
			if(intron) introns.add(nodes.get(n));
		}
		for(Node intron : introns)
			nodes.remove(intron);
	}

	public void addIntrons() {
		this.nodes.addAll(introns);
	}

	public Genome copy() {
		Genome genome = new Genome(random.nextInt());
		for(Node each : nodes)
			genome.nodes.add(each.copy());
		genome.parameters = new HashMap<String, Double>();
		for(String param : paramTypes)
			genome.parameters.put(param, parameters.get(param));
		return genome;
	}

	public void addNode(Node node) {
		nodes.add(node);
	}

	public void replaceNode(int index, Node node) {
		nodes.set(index, node);
	}

	public void crossover(Genome mate) {
		double crossoverProb = random.nextDouble();
		double addProb = parameters.get("crossover-add");
		if(mate.size() != 0)
			for(int i = 0; i < size(); i++)
				if (random.nextDouble() < crossoverProb)
					if (random.nextDouble() < addProb && nodes.size() < maxNodes)
						nodes.add(mate.nodes.get(random.nextInt(mate.size())));
					else
						nodes.set(i, mate.nodes.get(random.nextInt(mate.size())));
		for(String param : paramTypes)
			if(random.nextDouble() < crossoverProb)
				parameters.put(param, mate.parameters.get(param));

	}

	public int getNumIntrons() {
		return introns.size();
	}

	public double getAverageDepth() {
		double avgDepth = 0.0;
		for(Node node : nodes)
			avgDepth += node.getDepth();
		avgDepth /= nodes.size();
		return avgDepth;
	}

	public void updateNodeDistribution(Map<String, Double> nodeTypes) {
		for(Node node : nodes){
			String name = node.getNodeName();
			if(!nodeTypes.containsKey(name))
				nodeTypes.put(name, 1.0);
			else
				nodeTypes.put(name, nodeTypes.get(name) + 1.0);
		}
	}
	
	public void updateParameterStatistics(Map<String, Double> params) {
		for(String each : paramTypes){
			if(!params.containsKey(each))
				params.put(each, parameters.get(each));
			else
				params.put(each, params.get(each) + parameters.get(each));
		}
	}

	public void removeNode(int index) {
		this.nodes.remove(index);
	}

	public double getParameter(String parameter) {
		return this.parameters.get(parameter);
	}

	public void putParameter(String param, double value) {
		this.parameters.put(param, value);
	}

	public void setBeta(Matrix beta) {
		this.beta = beta;
	}
	
	public Matrix getBeta() {
		return beta;
	}

	public void setFitness(double fitness) {
		this.fitness = fitness;
	}
	
	public double getFitness() {
		return fitness;
	}

	@Override
	public int compareTo(Genome g2) {
		if(this.fitness == g2.fitness) return 0;
		return this.fitness < g2.fitness ? 1 : -1;
	}

}
