//~ corpus de funciones estilo zak
//~ prefijo: zak_ en funciones, variables locales con zak_

#include <stdio.h>
#include <string.h>
#include <math.h>

//~ 01 suma dos enteros
int zak_suma(int zak_a, int zak_b) {
    int zak_res = zak_a + zak_b;
    return zak_res;
}

//~ 02 resta dos enteros
int zak_resta(int zak_a, int zak_b) {
    int zak_res = zak_a - zak_b;
    return zak_res;
}

//~ 03 multiplica dos enteros
int zak_multi(int zak_a, int zak_b) {
    int zak_res = zak_a * zak_b;
    return zak_res;
}

//~ 04 divide dos flotantes
float zak_divide(float zak_a, float zak_b) {
    if (zak_b == 0.0f) return 0.0f;
    float zak_res = zak_a / zak_b;
    return zak_res;
}

//~ 05 valor absoluto entero
int zak_absoluto(int zak_n) {
    if (zak_n < 0) return -zak_n;
    return zak_n;
}

//~ 06 maximo de dos enteros
int zak_maximo(int zak_a, int zak_b) {
    if (zak_a > zak_b) return zak_a;
    return zak_b;
}

//~ 07 minimo de dos enteros
int zak_minimo(int zak_a, int zak_b) {
    if (zak_a < zak_b) return zak_a;
    return zak_b;
}

//~ 08 numero es par
int zak_es_par(int zak_n) {
    if (zak_n % 2 == 0) return 1;
    return 0;
}

//~ 09 numero es impar
int zak_es_impar(int zak_n) {
    if (zak_n % 2 != 0) return 1;
    return 0;
}

//~ 10 factorial iterativo
int zak_factorial(int zak_n) {
    int zak_res = 1;
    int zak_i = 1;
    while (zak_i <= zak_n) {
        zak_res = zak_res * zak_i;
        zak_i = zak_i + 1;
    }
    return zak_res;
}

//~ 11 potencia entera
int zak_potencia(int zak_base, int zak_exp) {
    int zak_res = 1;
    int zak_i = 0;
    while (zak_i < zak_exp) {
        zak_res = zak_res * zak_base;
        zak_i = zak_i + 1;
    }
    return zak_res;
}

//~ 12 suma de 1 hasta n
int zak_suma_n(int zak_n) {
    int zak_res = 0;
    int zak_i = 1;
    while (zak_i <= zak_n) {
        zak_res = zak_res + zak_i;
        zak_i = zak_i + 1;
    }
    return zak_res;
}

//~ 13 numero es positivo
int zak_es_positivo(int zak_n) {
    if (zak_n > 0) return 1;
    return 0;
}

//~ 14 numero es negativo
int zak_es_negativo(int zak_n) {
    if (zak_n < 0) return 1;
    return 0;
}

//~ 15 numero es cero
int zak_es_cero(int zak_n) {
    if (zak_n == 0) return 1;
    return 0;
}

//~ 16 intercambiar dos valores
void zak_swap(int *zak_a, int *zak_b) {
    int zak_tmp = *zak_a;
    *zak_a = *zak_b;
    *zak_b = zak_tmp;
}

//~ 17 longitud de cadena
int zak_longitud(char *zak_cad) {
    int zak_cont = 0;
    while (zak_cad[zak_cont] != '\0') {
        zak_cont = zak_cont + 1;
    }
    return zak_cont;
}

//~ 18 contar vocales en cadena
int zak_contar_vocales(char *zak_cad) {
    int zak_cont = 0;
    int zak_i = 0;
    while (zak_cad[zak_i] != '\0') {
        char zak_c = zak_cad[zak_i];
        if (zak_c=='a' || zak_c=='e' || zak_c=='i' || zak_c=='o' || zak_c=='u') {
            zak_cont = zak_cont + 1;
        }
        zak_i = zak_i + 1;
    }
    return zak_cont;
}

//~ 19 contar consonantes en cadena
int zak_contar_consonantes(char *zak_cad) {
    int zak_cont = 0;
    int zak_i = 0;
    while (zak_cad[zak_i] != '\0') {
        char zak_c = zak_cad[zak_i];
        if (zak_c>='a' && zak_c<='z') {
            if (zak_c!='a' && zak_c!='e' && zak_c!='i' && zak_c!='o' && zak_c!='u') {
                zak_cont = zak_cont + 1;
            }
        }
        zak_i = zak_i + 1;
    }
    return zak_cont;
}

//~ 20 invertir arreglo de enteros
void zak_invertir(int *zak_arr, int zak_n) {
    int zak_izq = 0;
    int zak_der = zak_n - 1;
    while (zak_izq < zak_der) {
        int zak_tmp = zak_arr[zak_izq];
        zak_arr[zak_izq] = zak_arr[zak_der];
        zak_arr[zak_der] = zak_tmp;
        zak_izq = zak_izq + 1;
        zak_der = zak_der - 1;
    }
}

//~ 21 busqueda lineal en arreglo
int zak_buscar(int *zak_arr, int zak_n, int zak_objetivo) {
    int zak_i = 0;
    while (zak_i < zak_n) {
        if (zak_arr[zak_i] == zak_objetivo) return zak_i;
        zak_i = zak_i + 1;
    }
    return -1;
}

//~ 22 suma de elementos de arreglo
int zak_suma_arr(int *zak_arr, int zak_n) {
    int zak_res = 0;
    int zak_i = 0;
    while (zak_i < zak_n) {
        zak_res = zak_res + zak_arr[zak_i];
        zak_i = zak_i + 1;
    }
    return zak_res;
}

//~ 23 promedio de arreglo
float zak_promedio(int *zak_arr, int zak_n) {
    if (zak_n == 0) return 0.0f;
    int zak_suma = zak_suma_arr(zak_arr, zak_n);
    float zak_res = (float)zak_suma / (float)zak_n;
    return zak_res;
}

//~ 24 maximo en arreglo
int zak_max_arr(int *zak_arr, int zak_n) {
    int zak_max = zak_arr[0];
    int zak_i = 1;
    while (zak_i < zak_n) {
        if (zak_arr[zak_i] > zak_max) zak_max = zak_arr[zak_i];
        zak_i = zak_i + 1;
    }
    return zak_max;
}

//~ 25 minimo en arreglo
int zak_min_arr(int *zak_arr, int zak_n) {
    int zak_min = zak_arr[0];
    int zak_i = 1;
    while (zak_i < zak_n) {
        if (zak_arr[zak_i] < zak_min) zak_min = zak_arr[zak_i];
        zak_i = zak_i + 1;
    }
    return zak_min;
}

//~ 26 numero es primo
int zak_es_primo(int zak_n) {
    if (zak_n < 2) return 0;
    int zak_i = 2;
    while (zak_i * zak_i <= zak_n) {
        if (zak_n % zak_i == 0) return 0;
        zak_i = zak_i + 1;
    }
    return 1;
}

//~ 27 contar primos hasta n
int zak_contar_primos(int zak_n) {
    int zak_cont = 0;
    int zak_i = 2;
    while (zak_i <= zak_n) {
        if (zak_es_primo(zak_i)) zak_cont = zak_cont + 1;
        zak_i = zak_i + 1;
    }
    return zak_cont;
}

//~ 28 maximo comun divisor
int zak_mcd(int zak_a, int zak_b) {
    while (zak_b != 0) {
        int zak_tmp = zak_b;
        zak_b = zak_a % zak_b;
        zak_a = zak_tmp;
    }
    return zak_a;
}

//~ 29 minimo comun multiplo
int zak_mcm(int zak_a, int zak_b) {
    int zak_res = (zak_a / zak_mcd(zak_a, zak_b)) * zak_b;
    return zak_res;
}

//~ 30 fibonacci iterativo
int zak_fibonacci(int zak_n) {
    if (zak_n <= 0) return 0;
    if (zak_n == 1) return 1;
    int zak_a = 0;
    int zak_b = 1;
    int zak_i = 2;
    while (zak_i <= zak_n) {
        int zak_tmp = zak_a + zak_b;
        zak_a = zak_b;
        zak_b = zak_tmp;
        zak_i = zak_i + 1;
    }
    return zak_b;
}

//~ 31 es numero perfecto
int zak_es_perfecto(int zak_n) {
    if (zak_n < 2) return 0;
    int zak_suma = 1;
    int zak_i = 2;
    while (zak_i * zak_i <= zak_n) {
        if (zak_n % zak_i == 0) {
            zak_suma = zak_suma + zak_i;
            if (zak_i != zak_n / zak_i) zak_suma = zak_suma + zak_n / zak_i;
        }
        zak_i = zak_i + 1;
    }
    if (zak_suma == zak_n) return 1;
    return 0;
}

//~ 32 contar digitos de numero
int zak_contar_digitos(int zak_n) {
    if (zak_n == 0) return 1;
    if (zak_n < 0) zak_n = -zak_n;
    int zak_cont = 0;
    while (zak_n > 0) {
        zak_n = zak_n / 10;
        zak_cont = zak_cont + 1;
    }
    return zak_cont;
}

//~ 33 suma de digitos
int zak_suma_digitos(int zak_n) {
    if (zak_n < 0) zak_n = -zak_n;
    int zak_res = 0;
    while (zak_n > 0) {
        zak_res = zak_res + (zak_n % 10);
        zak_n = zak_n / 10;
    }
    return zak_res;
}

//~ 34 numero es palindromo
int zak_es_palindromo_num(int zak_n) {
    if (zak_n < 0) return 0;
    int zak_orig = zak_n;
    int zak_rev = 0;
    while (zak_n > 0) {
        zak_rev = zak_rev * 10 + zak_n % 10;
        zak_n = zak_n / 10;
    }
    if (zak_orig == zak_rev) return 1;
    return 0;
}

//~ 35 invertir numero
int zak_invertir_num(int zak_n) {
    int zak_res = 0;
    while (zak_n > 0) {
        zak_res = zak_res * 10 + zak_n % 10;
        zak_n = zak_n / 10;
    }
    return zak_res;
}

//~ 36 cadena es palindromo
int zak_es_palindromo_cad(char *zak_cad) {
    int zak_len = zak_longitud(zak_cad);
    int zak_izq = 0;
    int zak_der = zak_len - 1;
    while (zak_izq < zak_der) {
        if (zak_cad[zak_izq] != zak_cad[zak_der]) return 0;
        zak_izq = zak_izq + 1;
        zak_der = zak_der - 1;
    }
    return 1;
}

//~ 37 convertir minusculas a mayusculas
void zak_a_mayusculas(char *zak_cad) {
    int zak_i = 0;
    while (zak_cad[zak_i] != '\0') {
        if (zak_cad[zak_i] >= 'a' && zak_cad[zak_i] <= 'z') {
            zak_cad[zak_i] = zak_cad[zak_i] - 32;
        }
        zak_i = zak_i + 1;
    }
}

//~ 38 convertir mayusculas a minusculas
void zak_a_minusculas(char *zak_cad) {
    int zak_i = 0;
    while (zak_cad[zak_i] != '\0') {
        if (zak_cad[zak_i] >= 'A' && zak_cad[zak_i] <= 'Z') {
            zak_cad[zak_i] = zak_cad[zak_i] + 32;
        }
        zak_i = zak_i + 1;
    }
}

//~ 39 contar ocurrencias de caracter
int zak_contar_char(char *zak_cad, char zak_c) {
    int zak_cont = 0;
    int zak_i = 0;
    while (zak_cad[zak_i] != '\0') {
        if (zak_cad[zak_i] == zak_c) zak_cont = zak_cont + 1;
        zak_i = zak_i + 1;
    }
    return zak_cont;
}

//~ 40 llenar arreglo con ceros
void zak_llenar_ceros(int *zak_arr, int zak_n) {
    int zak_i = 0;
    while (zak_i < zak_n) {
        zak_arr[zak_i] = 0;
        zak_i = zak_i + 1;
    }
}

//~ 41 copiar arreglo
void zak_copiar_arr(int *zak_src, int *zak_dst, int zak_n) {
    int zak_i = 0;
    while (zak_i < zak_n) {
        zak_dst[zak_i] = zak_src[zak_i];
        zak_i = zak_i + 1;
    }
}

//~ 42 arreglos son iguales
int zak_arr_iguales(int *zak_a, int *zak_b, int zak_n) {
    int zak_i = 0;
    while (zak_i < zak_n) {
        if (zak_a[zak_i] != zak_b[zak_i]) return 0;
        zak_i = zak_i + 1;
    }
    return 1;
}

//~ 43 bubble sort ascendente
void zak_burbuja(int *zak_arr, int zak_n) {
    int zak_i = 0;
    while (zak_i < zak_n - 1) {
        int zak_j = 0;
        while (zak_j < zak_n - zak_i - 1) {
            if (zak_arr[zak_j] > zak_arr[zak_j + 1]) {
                int zak_tmp = zak_arr[zak_j];
                zak_arr[zak_j] = zak_arr[zak_j + 1];
                zak_arr[zak_j + 1] = zak_tmp;
            }
            zak_j = zak_j + 1;
        }
        zak_i = zak_i + 1;
    }
}

//~ 44 busqueda binaria en arreglo ordenado
int zak_binaria(int *zak_arr, int zak_n, int zak_objetivo) {
    int zak_izq = 0;
    int zak_der = zak_n - 1;
    while (zak_izq <= zak_der) {
        int zak_mid = (zak_izq + zak_der) / 2;
        if (zak_arr[zak_mid] == zak_objetivo) return zak_mid;
        if (zak_arr[zak_mid] < zak_objetivo) zak_izq = zak_mid + 1;
        else zak_der = zak_mid - 1;
    }
    return -1;
}

//~ 45 contar pares en arreglo
int zak_contar_pares(int *zak_arr, int zak_n) {
    int zak_cont = 0;
    int zak_i = 0;
    while (zak_i < zak_n) {
        if (zak_arr[zak_i] % 2 == 0) zak_cont = zak_cont + 1;
        zak_i = zak_i + 1;
    }
    return zak_cont;
}

//~ 46 contar impares en arreglo
int zak_contar_impares(int *zak_arr, int zak_n) {
    int zak_cont = 0;
    int zak_i = 0;
    while (zak_i < zak_n) {
        if (zak_arr[zak_i] % 2 != 0) zak_cont = zak_cont + 1;
        zak_i = zak_i + 1;
    }
    return zak_cont;
}

//~ 47 contar negativos en arreglo
int zak_contar_negativos(int *zak_arr, int zak_n) {
    int zak_cont = 0;
    int zak_i = 0;
    while (zak_i < zak_n) {
        if (zak_arr[zak_i] < 0) zak_cont = zak_cont + 1;
        zak_i = zak_i + 1;
    }
    return zak_cont;
}

//~ 48 contar positivos en arreglo
int zak_contar_positivos(int *zak_arr, int zak_n) {
    int zak_cont = 0;
    int zak_i = 0;
    while (zak_i < zak_n) {
        if (zak_arr[zak_i] > 0) zak_cont = zak_cont + 1;
        zak_i = zak_i + 1;
    }
    return zak_cont;
}

//~ 49 imprimir arreglo
void zak_imprimir_arr(int *zak_arr, int zak_n) {
    int zak_i = 0;
    while (zak_i < zak_n) {
        printf("%d ", zak_arr[zak_i]);
        zak_i = zak_i + 1;
    }
    printf("\n");
}

//~ 50 es multiplo de k
int zak_es_multiplo(int zak_n, int zak_k) {
    if (zak_k == 0) return 0;
    if (zak_n % zak_k == 0) return 1;
    return 0;
}

//~ 51 contar multiplos de k hasta n
int zak_contar_multiplos(int zak_n, int zak_k) {
    if (zak_k == 0) return 0;
    int zak_cont = 0;
    int zak_i = 1;
    while (zak_i <= zak_n) {
        if (zak_i % zak_k == 0) zak_cont = zak_cont + 1;
        zak_i = zak_i + 1;
    }
    return zak_cont;
}

//~ 52 elevar flotante al cuadrado
float zak_cuadrado_f(float zak_n) {
    float zak_res = zak_n * zak_n;
    return zak_res;
}

//~ 53 elevar flotante al cubo
float zak_cubo_f(float zak_n) {
    float zak_res = zak_n * zak_n * zak_n;
    return zak_res;
}

//~ 54 clamp entero entre min y max
int zak_clamp(int zak_n, int zak_lo, int zak_hi) {
    if (zak_n < zak_lo) return zak_lo;
    if (zak_n > zak_hi) return zak_hi;
    return zak_n;
}

//~ 55 numero de divisores
int zak_num_divisores(int zak_n) {
    if (zak_n <= 0) return 0;
    int zak_cont = 0;
    int zak_i = 1;
    while (zak_i <= zak_n) {
        if (zak_n % zak_i == 0) zak_cont = zak_cont + 1;
        zak_i = zak_i + 1;
    }
    return zak_cont;
}

//~ 56 sumar digitos hasta un digito
int zak_raiz_digital(int zak_n) {
    while (zak_n >= 10) {
        zak_n = zak_suma_digitos(zak_n);
    }
    return zak_n;
}

//~ 57 es cuadrado perfecto
int zak_es_cuadrado(int zak_n) {
    if (zak_n < 0) return 0;
    int zak_i = 0;
    while (zak_i * zak_i <= zak_n) {
        if (zak_i * zak_i == zak_n) return 1;
        zak_i = zak_i + 1;
    }
    return 0;
}

//~ 58 producto de digitos
int zak_producto_digitos(int zak_n) {
    if (zak_n < 0) zak_n = -zak_n;
    if (zak_n == 0) return 0;
    int zak_res = 1;
    while (zak_n > 0) {
        zak_res = zak_res * (zak_n % 10);
        zak_n = zak_n / 10;
    }
    return zak_res;
}

//~ 59 contar espacios en cadena
int zak_contar_espacios(char *zak_cad) {
    int zak_cont = 0;
    int zak_i = 0;
    while (zak_cad[zak_i] != '\0') {
        if (zak_cad[zak_i] == ' ') zak_cont = zak_cont + 1;
        zak_i = zak_i + 1;
    }
    return zak_cont;
}

//~ 60 comparar dos cadenas
int zak_comparar(char *zak_a, char *zak_b) {
    int zak_i = 0;
    while (zak_a[zak_i] != '\0' && zak_b[zak_i] != '\0') {
        if (zak_a[zak_i] != zak_b[zak_i]) return 0;
        zak_i = zak_i + 1;
    }
    if (zak_a[zak_i] == '\0' && zak_b[zak_i] == '\0') return 1;
    return 0;
}

//~ 61 numero de palabras en cadena
int zak_contar_palabras(char *zak_cad) {
    int zak_cont = 0;
    int zak_dentro = 0;
    int zak_i = 0;
    while (zak_cad[zak_i] != '\0') {
        if (zak_cad[zak_i] != ' ' && !zak_dentro) {
            zak_dentro = 1;
            zak_cont = zak_cont + 1;
        } else if (zak_cad[zak_i] == ' ') {
            zak_dentro = 0;
        }
        zak_i = zak_i + 1;
    }
    return zak_cont;
}

//~ 62 primer digito de numero
int zak_primer_digito(int zak_n) {
    if (zak_n < 0) zak_n = -zak_n;
    while (zak_n >= 10) {
        zak_n = zak_n / 10;
    }
    return zak_n;
}

//~ 63 ultimo digito de numero
int zak_ultimo_digito(int zak_n) {
    if (zak_n < 0) zak_n = -zak_n;
    return zak_n % 10;
}

//~ 64 arreglo tiene elemento
int zak_tiene(int *zak_arr, int zak_n, int zak_val) {
    int zak_i = 0;
    while (zak_i < zak_n) {
        if (zak_arr[zak_i] == zak_val) return 1;
        zak_i = zak_i + 1;
    }
    return 0;
}

//~ 65 numero de ocurrencias en arreglo
int zak_ocurrencias(int *zak_arr, int zak_n, int zak_val) {
    int zak_cont = 0;
    int zak_i = 0;
    while (zak_i < zak_n) {
        if (zak_arr[zak_i] == zak_val) zak_cont = zak_cont + 1;
        zak_i = zak_i + 1;
    }
    return zak_cont;
}