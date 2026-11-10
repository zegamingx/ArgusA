package com.argus.agi.sensor

import java.net.InetAddress
import kotlinx.coroutines.test.runTest
import okhttp3.Dns
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ArticleExtractorSecurityTest {

    @Test
    fun rejectsPrivateAddressReturnedByTheActualHttpDnsResolver() = runTest {
        val privateDns = Dns {
            listOf(InetAddress.getByName("127.0.0.1"))
        }
        val extractor = ArticleExtractor(dns = privateDns)

        val error = runCatching {
            extractor.extract("https://example.org/article")
        }.exceptionOrNull()

        assertTrue(error is ArticleExtractionException)
        assertTrue(error?.message.orEmpty().contains("non-public address"))
    }

    @Test
    fun rejectsPrivateAndReservedIpv4Ranges() {
        val blockedAddresses = listOf(
            "0.0.0.1",
            "10.0.0.1",
            "100.64.0.1",
            "127.0.0.1",
            "169.254.1.1",
            "172.16.0.1",
            "192.0.0.1",
            "192.0.2.1",
            "192.168.1.1",
            "198.18.0.1",
            "198.51.100.1",
            "203.0.113.1",
            "224.0.0.1",
            "240.0.0.1"
        )

        blockedAddresses.forEach { address ->
            assertFalse(
                "Expected $address to be rejected",
                isPubliclyRoutableAddress(InetAddress.getByName(address))
            )
        }
    }

    @Test
    fun acceptsPublicIpv4Address() {
        assertTrue(
            isPubliclyRoutableAddress(InetAddress.getByName("93.184.216.34"))
        )
    }

    @Test
    fun rejectsIpv6UniqueLocalAndDocumentationRanges() {
        val blockedAddresses = listOf(
            "fc00::1",
            "fd12:3456:789a::1",
            "2001:db8::1",
            "fe80::1",
            "ff02::1"
        )

        blockedAddresses.forEach { address ->
            assertFalse(
                "Expected $address to be rejected",
                isPubliclyRoutableAddress(InetAddress.getByName(address))
            )
        }
    }
}
