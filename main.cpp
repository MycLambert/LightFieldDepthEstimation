#include "opencv2/highgui/highgui.hpp"//包含opencv库头文件
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2\opencv.hpp>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <omp.h>
#include <io.h>
#include <windows.h>  
#include <vector> 
#include <stdio.h>
#include <time.h>
using namespace cv;
using namespace std;

//计算图像梯度，用来判定清晰度
uint gradsFUN(Mat& finim);

//#1→#2图像与白图像相消，用来消除透镜、环境、光源等带来的误差
void cut(Mat& outputim, Mat& inputim);

//绘制直方图
Mat HisDraw(Mat &inpim,Mat &hist_img);

//计算模糊度
int VideoBlurDetect(Mat &srcimg);

void getFiles(string path, vector<string>& files);

void readTxt(string file);

double calculation_depth(string IM_name);

vector<string> string_split(const string &s, const string &seperator);

int main(int argc, char** argv) {//主函数

	system("dir /b /a-d D:\\data\\光场相机检测\\光场相机图像处理\\20180727\\*.tiff >D:\\data\\光场相机检测\\光场相机图像处理\\20180727\\allfiles.txt");
	readTxt("D:\\data\\光场相机检测\\光场相机图像处理\\20180727\\allfiles.txt");
	system("pause");

	return 0;
}


//计算图像梯度，用来判定清晰度
uint gradsFUN(Mat& gray)
{
	uint grads = 0, gradsplus = 0;
	// 读取图像  
	//Mat img = finim;

	// 转换为灰度图像  
	//Mat gray = finim1;
			 //cvtColor(img, gray, CV_BGR2GRAY);

			 // 求得x和y方向的一阶微分  
	Mat sobelx;
	Mat sobely;
	Sobel(gray, sobelx, CV_32F, 1, 0);
	Sobel(gray, sobely, CV_32F, 0, 1);

	// 求得梯度和方向  
	Mat norm;
	Mat dir;
	cartToPolar(sobelx, sobely, norm, dir);

	Mat sobel;
	sobel = abs(sobelx) + abs(sobely);

	// 转换为8位单通道图像进行显示  
	double normMax;
	minMaxLoc(norm, NULL, &normMax);
	Mat grad;
	norm.convertTo(grad, CV_8UC1, 255.0 / normMax, 0);

	double sobmin, sobmax;
	minMaxLoc(sobel, &sobmin, &sobmax);
	Mat gradplus;
	Mat sobelImage;
	sobel.convertTo(sobelImage, CV_8U, -255. / sobmax, 255);


	int i = 0;
	int j = 0;

	for (i = 0; i < grad.cols; i++)		
	{
		for (j = 0; j < grad.cols; j++)
		{
			grads += abs(grad.at<uchar>(j, i));
			if (grads > 6000)
			{
				//cout << grads << endl;
				gradsplus += abs(sobelImage.at<uchar>(j, i));
			}
		}
	}

	return gradsplus;
}

//#1→#2图像与白图像相消，用来消除透镜、环境、光源等带来的误差
void cut(Mat& outputim, Mat& inputim)
{
	for (int j = 0; j < inputim.cols; j++)
	{
		for (int i = 0; i < inputim.rows; i++)
		{
			outputim.at<uchar>(i, j) = -outputim.at<uchar>(i, j) + 150  + inputim.at<uchar>(i, j);
		}
	}

}

Mat HisDraw(Mat &inpim, Mat &hist_img)
{	
	Mat src, gray;
	src = inpim;
	gray = inpim;
	//cvtColor(src, gray, CV_RGB2GRAY);
	int bins = 256;
	int hist_size[] = { bins };
	float range[] = { 0, 256 };
	const float* ranges[] = { range };
	MatND hist;
	int channels[] = { 0 };

	calcHist(&gray, 1, channels, Mat(), // do not use mask  
		hist, 1, hist_size, ranges,
		true, // the histogram is uniform  
		false);

	double max_val;
	minMaxLoc(hist, 0, &max_val, 0, 0);
	int scale = 2;
	int hist_height = 256;
	hist_img = Mat::zeros(hist_height, bins*scale, CV_8UC3);
	for (int i = 0; i < bins; i++)
	{
		float bin_val = hist.at<float>(i);
		int intensity = cvRound(bin_val*hist_height / max_val);  //要绘制的高度  
		rectangle(hist_img, Point(i*scale, hist_height - 1),
			Point((i + 1)*scale - 1, hist_height - intensity),
			CV_RGB(255, 255, 255));
	}
	return hist_img;
}

/*检测模糊度
返回值为模糊度，值越大越模糊，越小越清晰，范围在0到几十，10以下相对较清晰，一般为5。
调用时可在外部设定一个阀值，具体阈值根据实际情况决定，返回值超过阀值当作是模糊图片。
算法所耗时间在1毫秒内
*/
int VideoBlurDetect(Mat &srcimg)
{
	cv::Mat img = srcimg;
	//cv::cvtColor(srcimg, img, CV_BGR2GRAY); // 将输入的图片转为灰度图，使用灰度图检测模糊度

											//图片每行字节数及高  
	int width = img.cols;
	int height = img.rows;
	ushort* sobelTable = new ushort[width*height];
	memset(sobelTable, 0, width*height * sizeof(ushort));

	int i, j, mul;
	//指向图像首地址  
	uchar* udata = img.data;
	for (i = 1, mul = i*width; i < height - 1; i++, mul += width)
		for (j = 1; j < width - 1; j++)

			sobelTable[mul + j] = abs(udata[mul + j - width - 1] + 2 * udata[mul + j - 1] + udata[mul + j - 1 + width] - \
				udata[mul + j + 1 - width] - 2 * udata[mul + j + 1] - udata[mul + j + width + 1]);

	for (i = 1, mul = i*width; i < height - 1; i++, mul += width)
		for (j = 1; j < width - 1; j++)
			if (sobelTable[mul + j] < 50 || sobelTable[mul + j] <= sobelTable[mul + j - 1] || \
				sobelTable[mul + j] <= sobelTable[mul + j + 1]) sobelTable[mul + j] = 0;

	int totLen = 0;
	int totCount = 1;

	uchar suddenThre = 50;
	uchar sameThre = 3;
	//遍历图片  
	for (i = 1, mul = i*width; i < height - 1; i++, mul += width)
	{
		for (j = 1; j < width - 1; j++)
		{
			if (sobelTable[mul + j])
			{
				int   count = 0;
				uchar tmpThre = 5;
				uchar max = udata[mul + j] > udata[mul + j - 1] ? 0 : 1;

				for (int t = j; t > 0; t--)
				{
					count++;
					if (abs(udata[mul + t] - udata[mul + t - 1]) > suddenThre)
						break;

					if (max && udata[mul + t] > udata[mul + t - 1])
						break;

					if (!max && udata[mul + t] < udata[mul + t - 1])
						break;

					int tmp = 0;
					for (int s = t; s > 0; s--)
					{
						if (abs(udata[mul + t] - udata[mul + s]) < sameThre)
						{
							tmp++;
							if (tmp > tmpThre) break;
						}
						else break;
					}

					if (tmp > tmpThre) break;
				}

				max = udata[mul + j] > udata[mul + j + 1] ? 0 : 1;

				for (int t = j; t < width; t++)
				{
					count++;
					if (abs(udata[mul + t] - udata[mul + t + 1]) > suddenThre)
						break;

					if (max && udata[mul + t] > udata[mul + t + 1])
						break;

					if (!max && udata[mul + t] < udata[mul + t + 1])
						break;

					int tmp = 0;
					for (int s = t; s < width; s++)
					{
						if (abs(udata[mul + t] - udata[mul + s]) < sameThre)
						{
							tmp++;
							if (tmp > tmpThre) break;
						}
						else break;
					}

					if (tmp > tmpThre) break;
				}
				count--;

				totCount++;
				totLen += count;
			}
		}
	}
	//模糊度
	float result = (float)totLen / totCount;
	delete[] sobelTable;
	sobelTable = NULL;

	return result;
}

void getFiles(string path, vector<string>& files)
{
	//文件句柄  
	long   hFile = 0;
	//文件信息  
	struct _finddata_t fileinfo;
	string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			//如果是目录,迭代之  
			//如果不是,加入列表  
			if ((fileinfo.attrib &  _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					getFiles(p.assign(path).append("\\").append(fileinfo.name), files);
			}
			else
			{
				files.push_back(p.assign(path).append("\\").append(fileinfo.name));
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}

void readTxt(string file)
{
	time_t first, second;
	first = time(NULL);

	ifstream infile;
	infile.open(file.data());   //将文件流对象与文件连接起来 
	assert(infile.is_open());   //若失败,则输出错误消息,并终止程序运行 

	ofstream oFilet;
	string result_name = "D:\\data\\光场相机检测\\光场相机图像处理\\20180727\\result2\\result.csv";
	oFilet.open(result_name, ios::out | ios::trunc);

	ofstream oFilet_log;
	string log_name = "D:\\data\\光场相机检测\\光场相机图像处理\\20180727\\result2\\log.txt";
	oFilet_log.open(log_name, ios::out | ios::trunc);

	string s;
	int i = 0;
	string lenth;
	double depth;
	while (getline(infile, s))
	{
		i++;
		string lenth = string_split(string_split(s, "-")[7], "_")[0];
		double depth = calculation_depth(s);
		second = time(NULL);
		char diff_time_string[100];
		int diff_time = difftime(second, first);
		if (diff_time > 60)
		{
			sprintf_s(diff_time_string, "%dm%ds", diff_time / 60, diff_time % 60);
		}
		else
		{
			sprintf_s(diff_time_string, "%ds",diff_time);
		}
		cout << i << " - Lenth is:" << lenth << "  ---- " << depth << ", during time is:" << diff_time_string << endl;
		oFilet << lenth << ',' << depth << endl;
		oFilet_log << i << " - Lenth is:" << lenth << "  ---- " << depth << ", during time is:" << diff_time_string << endl;
	}
	infile.close();             //关闭文件输入流 
	oFilet.close();
	oFilet_log.close();
}

double calculation_depth(string IM_name)
{
	double D_result = 0;
	int D_num = 0;

	double mintemp, maxtemp;
	double c = 0, c1 = 0;
	Point minLoc, maxLoc;
	//max_element
	Mat inpIM, modIM, resIM, gcaIM;//输入图像，待识别模式图像，用来保存互相关计算值的图像,梯度计算区域
	int gc[32];//用来储存3*3区域中的每个子区域的梯度
	int Gmax, GmaxX, GmaxY;

	const double nX = 2.5, nY = 2.5;//扩大系数（宏像素区域尺寸 = 扩大系数 * 匹配模板尺寸）
	const double coefX = 126.0 / 3, coefY = 72.8;
	const int startX = 60, startY = 46;
	const double shiftX = 5, shiftY = 5;
	char flag = 0;//用来标记每过一行，初始列值变化半个

	string IM_full_name = "D:\\data\\光场相机检测\\光场相机图像处理\\20180727\\" + IM_name;
	Mat srcIM = imread(IM_full_name, 0);
	//GaussianBlur(srcIM, srcIM, Size(3, 3), 2, 2);
	medianBlur(srcIM, srcIM, 3);

	//ofstream oFilet;
	//string EX_full_name = "D:\\data\\光场相机检测\\光场相机图像处理\\20180727\\excell\\" + IM_name + "-Nex.csv";
	//oFilet.open(EX_full_name, ios::out | ios::trunc);
	Mat rgbIM;
	Mat depthIM(srcIM.rows, srcIM.cols, CV_8UC3, Scalar::all(0));
	Mat bagIM = imread("D://data/光场相机检测/光场相机图像处理/test20170614/raw/BG_Raw.tiff", 0);
	//cvtColor(srcIM, rgbIM, CV_GRAY2RGB);	
	Mat tempIM(20, 20, CV_8UC1, Scalar::all(255));
	Mat mask(40, 40, CV_8UC1, Scalar::all(0));
	Mat tempmod, tempinp;
	Mat Hisim;

	//namedWindow("原始图片");

	while (1)
	{


		//cut(srcIM, bagIM) ;
		cvtColor(srcIM, rgbIM, CV_GRAY2RGB);



		//双自由度模式识别
		for (int j = 0; j < 60; j++)
		{
			for (int i = 0; i < 90; i++)
			{
				//RX = startX + i * coefX + flag*(coefX / 2.0), RY = startY + j * (coefY / 2.0)
				double RX = startX + i * coefX + flag*(coefX / 2.0), RY = startY + j * (coefY / 2.0);

				//将模板中心点向右移动一个宏像素
				double RX1 = startX + (i + 1) * coefX + flag *(coefX / 2.0), RY1 = startY + (j + 0) * (coefY / 2.0);
				int step = 3;//为每次移动步长
				double block = shiftX * (nX - 1);//为总移动次数
												 //rectangle(rgbIM, Point(RX - shiftX * nX, RY - shiftY * nY), Point(RX + shiftX * nX, RY + shiftY * nY), Scalar(220, 0, 0), 2, 8, 0);


#pragma omp parallel for schedule(static,4) 
				for (int ky = 0; ky <= block * 2.0; ky += step)
				{
					//int ky = 0;
					for (int kx = 0; kx <= block; kx += step)
					{
						//设置梯度计算区域，用来寻找最大梯度区域
						gcaIM = srcIM(Rect(RX + kx - shiftX, RY + ky - nY * shiftY, 2 * shiftX, 2 * shiftY));
						gc[4 * ky + kx] = gradsFUN(gcaIM);

					}
				}

				//计算最大梯度位置
				int len = sizeof(gc) / sizeof(int);
				Gmax = max_element(gc, gc + len) - &gc[0];
				GmaxX = Gmax % 4;
				GmaxY = Gmax / 4;

				//将一微透镜宏像素全部区域设置为待匹配图片inpIM
				inpIM = srcIM(Rect(RX1 - shiftX * nX, RY1 - GmaxY - 1, 2.0 * shiftX * nX, 2.0 * shiftY * 1));

				//将右下相邻区域宏像素 最大梯度位置 设置为匹配模板
				modIM = srcIM(Rect(RX - GmaxX + 3, RY - GmaxY - 1, 2 * shiftX, 2 * shiftY));

				//进行模式识别
				//imshow("放大前", modIM);
				resize(modIM, tempmod, Size(0, 0), 10, 10);
				resize(inpIM, tempinp, Size(0, 0), 10, 10);
				//imshow("放大后", modIM);
				matchTemplate(tempinp, tempmod, resIM, TM_SQDIFF);

				//提取模式识别系数最大最小值
				minMaxLoc(resIM, &mintemp, &maxtemp, &minLoc, &maxLoc);

				//公式中的D值
				double D_value = (RX1 - shiftX * nX + (double)minLoc.x / 10) - (RX - GmaxX + 3);

				//将待匹配图片用方框圈起来（白线）
				/*
				c1 = c;
				c = coefX + shiftX * nX - GmaxX - shiftX - (double)minLoc.x / 10;
				double color = double(gc[Gmax] - 17000) / 8000.0;
				if (color > 1)
					color = 1;
				if (color < 0)
					color = 0;
				double depth = double(c - 30) * 255 / 5;
				if (depth > 510)
					depth = 510;
				if (depth < 0)
					depth = 0;
				double green = color *(510 - depth), red = color * depth;
				*/
				//cout << gc[Gmax] << endl;

				//char str[100];;
				//sprintf_s(str, "P%d", D_value);
				 if (gc[Gmax] > 8000)
				{
					 D_num++;
					 D_result += D_value;
					 /*
					//oFilet << minLoc.x << endl;
					//oFilet << D_value << endl;
					//蓝色标记宏像素
					// rectangle(rgbIM, Point(RX - shiftX * nX, RY - shiftY * nY), Point(RX + shiftX * nX, RY + shiftY * nY),
					//	 Scalar((D_value -32)/(40-32)*255, 0, 0), 4, 8, 0);

					//绿色标记最大互相关处
					rectangle(rgbIM, Rect(RX1 - shiftX * nX + minLoc.x / 10, RY1 - GmaxY - 1 + minLoc.y / 10, 2 * shiftX, 2 * shiftY),
						Scalar(0, 250, 0), 1, 8, 0);

					//红色标记最大梯度处
					rectangle(rgbIM, Rect(RX - GmaxX + 3, RY - GmaxY - 1, 2 * shiftX, 2 * shiftY), Scalar(0, 0, 220), 1, 8, 0);
					sprintf_s(str, "D: %2.1f", D_value);

					//在图片上写文字
					if (D_value < 40)
						putText(rgbIM, str, Point(RX - GmaxX + 3, RY - GmaxY - 1), FONT_HERSHEY_SIMPLEX, 0.3,
							Scalar(155, 253, 150), 1, 8);
					else
						putText(rgbIM, str, Point(RX - GmaxX + 3, RY - GmaxY - 1), FONT_HERSHEY_SIMPLEX, 0.3,
							Scalar(55, 53, 250), 1, 8);

					//计算并写出模糊度
					Mat macIM = srcIM(Rect(RX - shiftX * nX, RY - shiftY * nY, 2 * shiftX * nX, 2 * shiftY * nY));

					//sprintf_s(str, " %d", VideoBlurDetect(macIM));
					putText(rgbIM, str, Point(RX - shiftX * nX + 1, RY + 2), CV_FONT_HERSHEY_SCRIPT_SIMPLEX, 0.3,
						Scalar(0, 253, 250), 1, 8);
						*/


				}
			}
			flag = 1 - flag;
		}
		//oFilet.close();

		//cout << "计算结束" << endl;
		return D_result / D_num;

		Mat showIM;
		resize(rgbIM, showIM, Size(0, 0), 5, 5);
		imshow("原始图片", showIM);
		HisDraw(srcIM, Hisim);
		//imshow("直方图", Hisim);
		//resize(modIM, modIM, Size(0, 0), 10, 10);
	}

	//imshow("匹配结果", resIM);
ESC:
	waitKey(10);
}

vector<string> string_split(const string &s, const string &seperator) {
	vector<string> result;
	typedef string::size_type string_size;
	string_size i = 0;

	while (i != s.size()) {
		//找到字符串中首个不等于分隔符的字母；
		int flag = 0;
		while (i != s.size() && flag == 0) {
			flag = 1;
			for (string_size x = 0; x < seperator.size(); ++x)
				if (s[i] == seperator[x]) {
					++i;
					flag = 0;
					break;
				}
		}

		//找到又一个分隔符，将两个分隔符之间的字符串取出；
		flag = 0;
		string_size j = i;
		while (j != s.size() && flag == 0) {
			for (string_size x = 0; x < seperator.size(); ++x)
				if (s[j] == seperator[x]) {
					flag = 1;
					break;
				}
			if (flag == 0)
				++j;
		}
		if (i != j) {
			result.push_back(s.substr(i, j - i));
			i = j;
		}
	}
	return result;
}