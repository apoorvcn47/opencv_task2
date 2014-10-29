/**
 * @file SURF_FlannMatcher
 * @brief SURF detector + descriptor + FLANN Matcher
 * @author A. Huaman
 */

#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/features2d.hpp"

using namespace cv;
using namespace std;

void readme();

/**
 * @function main
 * @brief Main function
 */
int main( int argc, char** argv )
{
    String outputDirectory = "../output/output";
  if( argc != 5 )
  { readme(); return -1; }

    Mat img_1 = imread( argv[1], CV_LOAD_IMAGE_COLOR );
    Mat img_2 = imread( argv[2], CV_LOAD_IMAGE_COLOR );
    Mat img_3 = imread( argv[3], CV_LOAD_IMAGE_COLOR );
    Mat img_4 = imread( argv[4], CV_LOAD_IMAGE_COLOR );

  if( !img_1.data || !img_2.data || !img_3.data || !img_4.data)
  { cout<< " --(!) Error reading images " << endl; return -1; }

  //-- Step 1: Detect the keypoints using SURF Detector
  int minHessian = 400;     //comes from experience

  SurfFeatureDetector detector( minHessian );

  std::vector<KeyPoint> keypoints_1, keypoints_2, keypoints_3, keypoints_4;
    //kepoints are 2D features that are impacted by diffferent parameters of image(scale, orientation)
  detector.detect( img_1, keypoints_1 );
  detector.detect( img_2, keypoints_2 );
  detector.detect( img_3, keypoints_3 );
  detector.detect( img_4, keypoints_4 );

  //-- Step 2: Calculate descriptors (feature vectors)
  SurfDescriptorExtractor extractor;

  Mat descriptors_1, descriptors_2, descriptors_3, descriptors_4;
//descriptor are calculated by an alrithm that checks the neighbourhood of keypoints and provide descriptor
  extractor.compute( img_1, keypoints_1, descriptors_1 );
  extractor.compute( img_2, keypoints_2, descriptors_2 );
  extractor.compute( img_3, keypoints_3, descriptors_3 );
  extractor.compute( img_4, keypoints_4, descriptors_4 );

  //-- Step 3: Matching descriptor vectors using FLANN matcher
  FlannBasedMatcher matcher;
  std::vector< DMatch > matches12, matches13, matches14;
  matcher.match( descriptors_1, descriptors_2, matches12 );
  matcher.match( descriptors_1, descriptors_3, matches13 );
  matcher.match( descriptors_1, descriptors_4, matches14 );

  double max_dist12, max_dist13, max_dist14, min_dist12 , min_dist13, min_dist14;
  max_dist12= max_dist13= max_dist14=0;
  min_dist12 = min_dist13 = min_dist14 = 100;
  //-- Quick calculation of max and min distances between keypoints
  for( int i = 0; i < descriptors_1.rows; i++ )
  { double dist = matches12[i].distance;
    if( dist < min_dist12 ) min_dist12 = dist;
    if( dist > max_dist12 ) max_dist12 = dist;

    dist = matches13[i].distance;
        if( dist < min_dist13 ) min_dist13 = dist;
        if( dist > max_dist13 ) max_dist13 = dist;

    dist = matches14[i].distance;
        if( dist < min_dist14 ) min_dist14 = dist;
        if( dist > max_dist14 ) max_dist14 = dist;


  }



  printf("-- Max dist for 1&2: %f \n", max_dist12 );
  printf("-- Min dist for 1&2: %f \n", min_dist12 );

  printf("-- Max dist for 1&3: %f \n", max_dist13 );
  printf("-- Min dist for 1&3: %f \n", min_dist13 );

  printf("-- Max dist for 1&4: %f \n", max_dist14 );
  printf("-- Min dist for 1&4: %f \n", min_dist14 );

  //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
  //-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
  //-- small)
  //-- PS.- radiusMatch can also be used here.
  std::vector< DMatch > good_matches12, good_matches13, good_matches14;

  for(int i = 0; i < descriptors_1.rows; i++ )
  { if( matches12[i].distance <= max(2*min_dist12, 0.02) )  //0.02 comes from experience
    { good_matches12.push_back( matches12[i]); }

    if( matches13[i].distance <= max(2*min_dist13, 0.02) )
    { good_matches13.push_back( matches13[i]); }

    if( matches14[i].distance <= max(2*min_dist14, 0.02) )
    { good_matches14.push_back( matches14[i]); }
  }



  //-- Draw only "good" matches
  Mat img_matches12, img_matches13, img_matches14;
  drawMatches( img_1, keypoints_1, img_2, keypoints_2,
               good_matches12, img_matches12, Scalar::all(-1), Scalar::all(-1),
               vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

  drawMatches( img_1, keypoints_1, img_3, keypoints_3,
               good_matches13, img_matches13, Scalar::all(-1), Scalar::all(-1),
               vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

  drawMatches( img_1, keypoints_1, img_4, keypoints_4,
               good_matches14, img_matches14, Scalar::all(-1), Scalar::all(-1),
               vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

  //-- Show detected matches
//  imshow( "Good Matches", img_matches12 );

  imwrite(outputDirectory.append("2.jpg"),img_matches12);


//  imshow( "Good Matches", img_matches13 );

  imwrite(outputDirectory.append("3.jpg"),img_matches13);

//  imshow( "Good Matches", img_matches14 );

  imwrite(outputDirectory.append("4.jpg"),img_matches14);

//  for( int i = 0; i < (int)good_matches12.size(); i++ )
//  { printf( "-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, good_matches12[i].queryIdx, good_matches12[i].trainIdx ); }

  waitKey(0);

  return 0;
}

/**
 * @function readme
 */
void readme()
{ std::cout << " Usage: ./SURF_FlannMatcher <img1> <img2> <img3> <img4>" << std::endl; }
