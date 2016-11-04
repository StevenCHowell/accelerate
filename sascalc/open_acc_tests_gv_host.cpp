#include <math.h>
#include <cstdlib>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <ctime>

//////////////////////////////////////////////////////////
/// Set the golden vectors
//////////////////////////////////////////////////////////
double * getGV(int Ngv)
{
    // assign Ngv
    Ngv = Ngv;

    // setup GV
    if (Ngv%2==0)
    {
        std::cout<<"The number of golden vectors should be an odd integer, and so it will be reset to be: "<<Ngv+1<<std::endl;
        ++Ngv;
    }
    double *gv = new double[3*Ngv];
    
    const double phi_inv = 2.0/(1+sqrt(5.)); // golden ratio
    double cos_theta, sin_theta, phi;
    int igv;
    const int rank = Ngv/2;
    for (int i=-rank; i<=rank; i++)
    {   
        sin_theta = cos(asin(2.0*i/Ngv));
        cos_theta = 2.0*i/Ngv;
        phi = 2*M_PI*i*phi_inv;
        igv = i + rank;
        gv[igv] = sin_theta*cos(phi);
        gv[Ngv+igv] = sin_theta*sin(phi);
        gv[2*Ngv+igv] = cos_theta;
    }

    // return
    return gv;
}

//////////////////////////////////////////////////////////
/// Golden vector internal calculator
//  for single item and single frame using fixed method
//////////////////////////////////////////////////////////
double * calcIq(const double *const coor, const double *const B, const int Natoms, const int Nq, const double dQ, const int Ngv)
{
    // locals
    int iatom,iq,igv;
    double Ireal, Iimag;
    double qmag;
    double qx, qy, qz ;
    double *gv = getGV(Ngv);
    double *Iq = new double[Nq];

    clock_t begin = clock();

    // summation
    for (iq=0; iq<Nq; ++iq)
    {
        qmag = iq*dQ;
        for (igv=0; igv<Ngv; ++igv)
        {
            qx = qmag * gv[igv];
            qy = qmag * gv[Ngv+igv];
            qz = qmag * gv[2*Ngv+igv];
            Ireal = 0.0;
            Iimag = 0.0;
            for (iatom=0; iatom<Natoms; ++iatom)
            {
                Iimag += B[iatom]*sin(qx*coor[iatom]+qy*coor[Natoms+iatom]+qz*coor[Natoms*2+iatom]);
                Ireal += B[iatom]*cos(qx*coor[iatom]+qy*coor[Natoms+iatom]+qz*coor[Natoms*2+iatom]);
            }
            Iq[iq] += Ireal*Ireal + Iimag*Iimag;
        }
        Iq[iq] /= Ngv;
    }

    // clean up
    delete [] gv;
    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    printf("TIME = %f\n",elapsed_secs) ;
    // return
    return Iq;
}


//////////////////////////////////////////////////////////
/// main
//////////////////////////////////////////////////////////
int main()
{
    // create dummy coordinates and Bs
    // which is a linear molecule along X-axis with 10 atoms
    const int Natoms=1000000;
    double * coor = new double[Natoms*3];
    double * B = new double[Natoms];
    for (int iatom=0; iatom<Natoms; ++iatom)
    {
        coor[iatom] = iatom;
        coor[Natoms + iatom] = 0.0;
        coor[2*Natoms + iatom] = 0.0;

        B[iatom] = 1.0;
    }

    // Iq stuff
    const int Nq = 10;
    const double dQ = 0.1;
    const int Ngv = 31;

    // calculate Iq
    double * Iq = calcIq(coor, B, Natoms, Nq, dQ, Ngv);

    // print Iq
    printf("       q       Iq\n");
    for (int iq = 0; iq<Nq; ++iq)
    {
        printf("%8.3f %8.3f %8.3f\n", iq*dQ, Iq[iq], Iq[iq]/Iq[0]);
    }

    // clean up
    delete [] coor;
    delete [] B;
    return 0;
}
