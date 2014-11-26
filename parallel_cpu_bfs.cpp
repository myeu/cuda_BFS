#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
	if (argc != 3)
    {
		printf("Usage: %s <file> <num threads>\n",
               argv[0]);
		exit(EXIT_FAILURE);
	}

	int thread_count = atoi(argv[2]);
    char *filename = argv[1];
	FILE *fp = NULL;
    
	if ((fp = fopen(filename, "r")) == NULL)
    {
		printf("Can't open file %s\n", argv[1]);
		exit(EXIT_FAILURE);
	}
    
    if (thread_count < 1)
    {
        printf("The thread count must be at least 1\n");
        exit(EXIT_FAILURE);
    }

    printf("hi\n");

	return 0;
}