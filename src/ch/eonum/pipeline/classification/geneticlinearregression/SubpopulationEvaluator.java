package ch.eonum.pipeline.classification.geneticlinearregression;

import java.util.List;

import ch.eonum.pipeline.core.DataSet;
import ch.eonum.pipeline.core.Instance;

public class SubpopulationEvaluator<E extends Instance> implements Runnable {

	private int population;
	private List<Genome> genomes;
	private GeneticLinearRegression<E> parent;
	private DataSet<E> trainData;
	private DataSet<E> fitnessData;
	private double max;
	private int generation;
	private Genome bestGenome;

	public SubpopulationEvaluator(int population, int generation,
			List<Genome> genomes, GeneticLinearRegression<E> parent,
			DataSet<E> trainData, DataSet<E> fitnessData) {
		this.population = population;
		this.generation = generation;
		this.genomes = genomes;
		this.parent = parent;
		this.trainData = trainData;
		this.fitnessData = fitnessData;
	}

	public void selectBestIndividual() {
		// make a deep copy of the test data, hence we can evaluate concurrently
		DataSet<E> fitnessDataCopy = null;
		synchronized(fitnessData) {
			fitnessDataCopy = fitnessData.deepCopy();
		}
		double[][] trainMatrix = trainData.asDoubleArrayMatrix(parent
				.getFeatures());
		double[] outcomesTraining = trainData.outComesAsArray();
		max = Double.NEGATIVE_INFINITY;
		int maxI = -1;
		for(int i = 0; i < genomes.size(); i++){
			genomes.get(i).removeIntrons(trainMatrix);
			double eval = parent.evaluate(genomes.get(i), generation,
					population, i, trainMatrix, outcomesTraining,
					fitnessDataCopy, false);
			if(eval > max){
				max = eval; 
				maxI = i;
			}
		}
		parent.log("Best individual from population " + population + ": "
				+ maxI + " Fitness: " + max);
		bestGenome = genomes.get(maxI);
		bestGenome.setFitness(max);
	}

	@Override
	public void run() {
		this.selectBestIndividual();
	}

	public Genome getBestGenome() {
		return bestGenome;
	}

}
