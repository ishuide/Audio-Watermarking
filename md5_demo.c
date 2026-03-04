/*
 * MD5 Hashing - Avalanche Effect Demo
 * 
 * This program demonstrates that changing just a single bit in the input 
 * results in a completely different hash value.
 *
 * To compile: gcc md5_demo.c -o md5_demo -lcrypto
 */

#include <stdio.h>
#include <string.h>
#include <openssl/md5.h>

void print_hash(unsigned char *hash) {
    for(int i = 0; i < MD5_DIGEST_LENGTH; i++) printf("%02x", hash[i]);
    printf("\n");
}

int main() {
    char *input1 = "Hello Watermark";
    char *input2 = "Hello watermark"; // Only 'W' changed to 'w'
    
    unsigned char hash1[MD5_DIGEST_LENGTH];
    unsigned char hash2[MD5_DIGEST_LENGTH];

    MD5((unsigned char*)input1, strlen(input1), hash1);
    MD5((unsigned char*)input2, strlen(input2), hash2);

    printf("Input 1: %s\n", input1);
    printf("Hash 1 : "); print_hash(hash1);
    
    printf("\nInput 2: %s\n", input2);
    printf("Hash 2 : "); print_hash(hash2);

    printf("\nConclusion: A tiny change (W -> w) caused a massive change in the hash!\n");

    return 0;
}
