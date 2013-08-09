import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;

import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.supervised.instance.Resample;

public class Main {

	public static void main(String[] args) 
	{ //args[0] is path to the arff data file
		//args[1] is the number of times to resample the data
		//args[2] is the output file name.  It must be a .dot file.
		
		int resampleTimes = Integer.parseInt(args[1]);
		//read in the file
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
		//build trees based on resampled data and save them in an array
		J48[] trees = new J48[resampleTimes+1];
		//first build the tree with the original data
		try {
			trees[0] = new J48();
			trees[0].buildClassifier(instances);
		} catch (Exception e1) {
			e1.printStackTrace();
		}
		//resample data and build more trees
		for(int j = 1; j <= resampleTimes; j++)
		{
			Resample filter = new Resample();
			try{
				filter.setRandomSeed(j);
				filter.setInputFormat(instances);
				//need to input all instances before getting output
				for (int i = 0; i < instances.numInstances(); i++) 
				{
					filter.input(instances.instance(i));
				}
				filter.batchFinished();
			}
			catch(Exception e)
			{
				e.printStackTrace();
			}
			Instances newData = filter.getOutputFormat();
			Instance processed;
			while ((processed = filter.output()) != null) 
			{
			  newData.add(processed);
			}
			newData.setClassIndex(instances.numAttributes() - 1); 
			//build the tree with the resampled data
			try {
				trees[j] = new J48();
				trees[j].buildClassifier(newData);
			} catch (Exception e1) {
				e1.printStackTrace();
			}
		}
		
		//find the best tree using the original data as testing data
//		int[] correctCounts = new int[trees.length];
//		for(int i = 0; i < trees.length; i++)
//		{
//			for(int j = 0; j < instances.numInstances(); j++)
//			{
//				double classPrediction;
//				try {
//					classPrediction = trees[i].classifyInstance(instances.instance(j));
//				} catch (Exception e) {
//					e.printStackTrace();
//					return;
//				}
//				int indexOfClass = instances.instance(j).classIndex();
//				if(Double.parseDouble(instances.instance(j).stringValue(indexOfClass)) == classPrediction)
//				{
//					correctCounts[i]++;
//				}
//			}
//		}
//		J48 best = null;
//		int maxCorrect = 0;
//		for(int i = 0; i < correctCounts.length; i++)
//		{
//			if(correctCounts[i] > maxCorrect)
//			{
//				maxCorrect = correctCounts[i];
//				best = (J48)trees[i];
//			}
//		}
		//find smallest tree
		J48 best = null;
		double smallestSize = Double.MAX_VALUE;
		for(int i = 0; i < trees.length; i++)
		{
			if(trees[i].getMeasure("measureTreeSize") < smallestSize)
			{
				best = trees[i];
				smallestSize = trees[i].getMeasure("measureTreeSize");
			}
		}
//		Enumeration en = trees[0].enumerateMeasures();
//		while(en.hasMoreElements())
//		{
//			System.out.println(en.nextElement().toString());
//		}
		//display results
		String graph = null;
		try {
			graph = best.graph();
		} catch (Exception e1) {
			e1.printStackTrace();
		}
		try {
		      File file = new File(args[2]);
		      if (file.createNewFile())
		      {
		    	  PrintWriter out = new PrintWriter(args[2]);
		    	  out.println(graph);
		    	  out.close();
		      }
		      else
		      {
		        System.out.println("File already exists.");
		      }
	    } catch (IOException e) {
		      e.printStackTrace();
		}
		//System.out.println("Fraction correctly classified: "+((double)maxCorrect/instances.numInstances()));
	}
}
