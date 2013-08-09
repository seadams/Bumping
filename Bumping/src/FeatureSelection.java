import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;

import weka.attributeSelection.ConsistencySubsetEval;
import weka.attributeSelection.GreedyStepwise;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.supervised.instance.Resample;

public class FeatureSelection {

	public static void main(String[] args) {
		// args[0] is path to the arff data file
		// args[1] is the number of times to resample the data
		// args[2] is the output file name.

		int resampleTimes = Integer.parseInt(args[1]);
		// read in the file
		DataSource source;
		try {
			source = new DataSource(args[0]);
		} catch (Exception e1) {
			e1.printStackTrace();
			return;
		}
		Instances instances;
		try {
			instances = source.getDataSet();
		} catch (Exception e1) {
			e1.printStackTrace();
			return;
		}
		instances.setClassIndex(instances.numAttributes() - 1);
		// select attributes based on resampled data and save results in an
		// array
		GreedyStepwise searcher = new GreedyStepwise();
		int[] bestSubsetIndices = null;
		// first select attributes with the original data
		try {
			ConsistencySubsetEval eval = new ConsistencySubsetEval();
			eval.buildEvaluator(instances);
			int[] subsetIndices = searcher.search(eval, instances);
			bestSubsetIndices = subsetIndices;
		} catch (Exception e1) {
			e1.printStackTrace();
			return;
		}
		// resample data and select more subsets
		for (int j = 1; j <= resampleTimes; j++) 
		{
			Resample filter = new Resample();
			try {
				filter.setRandomSeed(j);
				filter.setInputFormat(instances);
				// need to input all instances before getting output
				for (int i = 0; i < instances.numInstances(); i++) 
				{
					filter.input(instances.instance(i));
				}
				filter.batchFinished();
			} catch (Exception e) {
				e.printStackTrace();
			}
			Instances newData = filter.getOutputFormat();
			Instance processed;
			while ((processed = filter.output()) != null) 
			{
				newData.add(processed);
			}
			newData.setClassIndex(instances.numAttributes() - 1);
			// select the attributes with the resampled data
			try {
				ConsistencySubsetEval eval = new ConsistencySubsetEval();
				eval.buildEvaluator(newData);
				int[] subsetIndices = searcher.search(eval, newData);
				if (subsetIndices.length < bestSubsetIndices.length) 
				{
					bestSubsetIndices = subsetIndices;
				}
			} catch (Exception e1) {
				e1.printStackTrace();
			}
		}
		// print out the selected attributes
		try {
			File file = new File(args[2]);
			if (file.createNewFile()) 
			{
				PrintWriter out = new PrintWriter(args[2]);
				for (int i = 0; i < bestSubsetIndices.length; i++) 
				{
					out.println(instances.attribute(bestSubsetIndices[i]).name());
				}
				out.close();
			} else 
			{
				System.out.println("File already exists.");
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

}
